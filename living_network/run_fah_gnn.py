"""Train one GNN on one frozen Fah task; reuse saved tree/LOMAP predictions."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import pickle
import random
import sqlite3
from urllib.parse import quote

import numpy as np
import pandas as pd
from rdkit import rdBase
from scipy.stats import spearmanr
import torch
from torch.nn import functional as F

from fah_gnn_models import FahGNN, collate_graphs, sphere, hybrid
from prepare_fah_gnn import graph_key, graph_recipe, keys_for
from rbfe_artifacts import file_hash, LOMAP_SCORING_VERSION
from run_fah_benchmark import atomic_json, metrics, bootstrap


CONFIG = {
    "seed": 42, "hidden_dim": 128, "embed_dim": 128, "num_layers": 3,
    "dropout": .1, "epochs": 60, "patience": 10, "batch_size": 64,
    "learning_rate": .001, "weight_decay": 1e-5, "gradient_clip": 5.,
    "hypersphere_contrastive_weight": 1.,
}


def request_hash(request):
    return hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()


def atomic_checkpoint(path, value):
    temporary = path.with_suffix(".partial")
    torch.save(value, temporary)
    temporary.replace(path)


def load_graph_cache(directory, frame, architecture, benchmark_hash):
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["request"]["recipe"] != graph_recipe(architecture):
        raise ValueError("Graph recipe/version differs; prepare a fresh cache")
    if benchmark_hash not in manifest["request"]["benchmark_hashes"]:
        raise ValueError("Graph cache does not include this benchmark")
    database = directory / "graphs.sqlite"
    if file_hash(database) != manifest["database_sha256"]:
        raise ValueError("Graph database checksum mismatch")
    connection = sqlite3.connect(f"file:{quote(str(database.resolve()))}?mode=ro", uri=True)
    needed = keys_for(frame, architecture)
    try:
        stored = connection.execute("SELECT value FROM metadata WHERE key='request'").fetchone()
        if stored is None or json.loads(stored[0]) != manifest["request"]:
            raise ValueError("Graph database request disagrees with manifest")
        graphs = {}
        for key, blob in connection.execute("SELECT key,data FROM graphs"):
            if key in needed:
                graphs[key] = pickle.loads(blob)
    finally:
        connection.close()
    if set(graphs) != needed:
        raise ValueError("Missing keyed graphs; row-position fallback is forbidden")
    return graphs, file_hash(manifest_path)


def graph_records(frame, graphs, architecture):
    if architecture == "hypersphere":
        return [(graphs[a], graphs[b]) for a, b in zip(frame.smi_a, frame.smi_b)]
    return [graphs[graph_key(a, b)] for a, b in zip(frame.smi_a, frame.smi_b)]


def graph_validity(frame, graphs, architecture):
    if architecture == "hypersphere":
        return np.array([graphs[a] is not None and graphs[b] is not None
                         for a, b in zip(frame.smi_a, frame.smi_b)])
    return np.array([graphs[graph_key(a, b)] is not None for a, b in zip(frame.smi_a, frame.smi_b)])


def training_statistics(train, pilot):
    log_y = np.log(train.target.to_numpy(dtype=float))
    stats = {"log_target_mean": float(log_y.mean()),
             "log_target_std": max(float(log_y.std()), 1e-6),
             "target_q99": max(float(np.quantile(train.target, .99)), 1e-8)}
    if pilot:
        log_p = np.log(train.pilot_proxy.to_numpy(dtype=float))
        stats.update(log_pilot_mean=float(log_p.mean()),
                     log_pilot_std=max(float(log_p.std()), 1e-6))
    return stats


def make_model(architecture, pilot, config, device):
    return FahGNN(architecture, pilot, **{
        key: config[key] for key in ("hidden_dim", "embed_dim", "num_layers", "dropout")
    }).to(device)


def batch_inputs(frame, graphs, architecture, pilot, stats, device):
    batch = collate_graphs(graph_records(frame, graphs, architecture), architecture, device)
    p = None
    if pilot:
        values = (np.log(frame.pilot_proxy.to_numpy(dtype=float)) - stats["log_pilot_mean"]) / stats["log_pilot_std"]
        p = torch.as_tensor(values, dtype=torch.float32, device=device)
    return batch, p


def predict_log(model, frame, graphs, architecture, pilot, stats, device, batch_size):
    """Preserve exact input order; never iterate unordered graph dictionaries."""
    model.eval()
    values = []
    with torch.no_grad():
        for start in range(0, len(frame), batch_size):
            part = frame.iloc[start:start + batch_size]
            graph, p = batch_inputs(part, graphs, architecture, pilot, stats, device)
            z, _ = model(graph, p)
            values.extend(z.cpu().double().numpy() * stats["log_target_std"] + stats["log_target_mean"])
    result = np.array(values)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite GNN predictions")
    return result


def train_model(train, val, graphs, architecture, pilot, output, identity, device, config=CONFIG):
    """Only training and validation frames enter training/checkpoint selection."""
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config["seed"])
    model = make_model(architecture, pilot, config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])
    stats = training_statistics(train, pilot)
    state_path = output / "checkpoint.pt"
    state = None
    if state_path.exists():
        state = torch.load(state_path, map_location=device, weights_only=False)
        if state["identity"] != identity or state["statistics"] != stats or state["config"] != config:
            raise ValueError("Checkpoint identity, training statistics, or configuration mismatch")
        if state["finished"]:
            print("Training already complete; preserving selected checkpoint.", flush=True)
            return state
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["rng_cpu"].cpu())
        if device.type == "cuda" and state["rng_cuda"] is not None:
            if len(state["rng_cuda"]) != torch.cuda.device_count():
                raise ValueError("CUDA device count changed during resume")
            torch.cuda.set_rng_state_all([x.cpu() for x in state["rng_cuda"]])
    start = state["epoch"] + 1 if state else 0
    best = state["best_spearman"] if state else -np.inf
    best_state = state["best_model"] if state else None
    best_epoch = state["best_epoch"] if state else None
    stale = state["stale_epochs"] if state else 0
    history = state["history"] if state else []
    for epoch in range(start, config["epochs"]):
        model.train()
        # Epoch-specific permutations reproduce interrupted runs without loader state.
        order = np.random.default_rng(config["seed"] + epoch).permutation(len(train))
        loss_sum = regression_sum = contrast_sum = 0.
        for begin in range(0, len(order), config["batch_size"]):
            part = train.iloc[order[begin:begin + config["batch_size"]]]
            batch, p = batch_inputs(part, graphs, architecture, pilot, stats, device)
            raw_y = torch.as_tensor(part.target.to_numpy(dtype=float), dtype=torch.float32, device=device)
            y = (raw_y.log() - stats["log_target_mean"]) / stats["log_target_std"]
            prediction, cosine = model(batch, p)
            regression = F.mse_loss(prediction, y)
            contrast = prediction.new_tensor(0.)
            if architecture == "hypersphere":
                target_cos = torch.cos(torch.pi * (raw_y / stats["target_q99"]).clamp(0, 1))
                contrast = F.mse_loss(cosine, target_cos)
            loss = regression + config["hypersphere_contrastive_weight"] * contrast
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(part)
            regression_sum += float(regression.detach()) * len(part)
            contrast_sum += float(contrast.detach()) * len(part)
        predicted = predict_log(model, val, graphs, architecture, pilot, stats, device, config["batch_size"])
        rho = float(spearmanr(val.target, predicted).statistic)
        improved = np.isfinite(rho) and rho > best
        if improved:
            best, best_epoch, stale = rho, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        history.append({"epoch": epoch + 1, "loss": loss_sum / len(train),
                        "regression_loss": regression_sum / len(train),
                        "contrastive_loss": contrast_sum / len(train),
                        "validation_spearman": rho if np.isfinite(rho) else None})
        finished = epoch + 1 == config["epochs"] or stale >= config["patience"]
        state = {
            "identity": identity, "config": config, "statistics": stats,
            "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "best_model": best_state, "best_epoch": best_epoch, "best_spearman": best,
            "stale_epochs": stale, "history": history, "finished": finished,
            "rng_cpu": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
            "training_device": str(device),
        }
        atomic_checkpoint(state_path, state)
        atomic_json(output / "history.json", history)
        print(f"Epoch {epoch + 1}: loss={loss_sum / len(train):.5f}, val rho={rho:.5f}, best={best:.5f}", flush=True)
        if finished:
            break
    if state is None or state["best_model"] is None:
        raise ValueError("No checkpoint with a defined validation correlation")
    return state


def align_reference(test, reference):
    if not test.edge_id.is_unique or not reference.edge_id.is_unique:
        raise ValueError("Repeated edge IDs in benchmark or reference predictions")
    if set(test.edge_id) != set(reference.edge_id):
        raise ValueError("Reference prediction cohort differs from the frozen test cohort")
    reference = reference.set_index("edge_id").loc[test.edge_id].reset_index()
    for name in ("smi_a", "smi_b", "split", "source_sha256", "resample_group_a", "resample_group_b"):
        if not np.array_equal(reference[name].to_numpy(), test[name].to_numpy()):
            raise ValueError(f"Reference identity differs: {name}")
    if not np.allclose(reference.target, test.target, rtol=1e-12, atol=1e-12):
        raise ValueError("Reference targets differ")
    if "pilot_proxy" in test and not np.allclose(reference.pilot_direct, test.pilot_proxy, rtol=1e-12, atol=1e-12):
        raise ValueError("Reference pilot inputs differ")
    return reference


def evaluate(test, valid, prediction, reference, architecture, pilot):
    reference = align_reference(test, reference)
    output = test.copy().reset_index(drop=True)
    output["graph_valid"] = valid
    output["gnn"] = np.nan
    output.loc[valid, "gnn"] = prediction
    names = {"structure": "tree_structure", "lomap": "lomap", "training_mean": "training_mean"}
    if "pilot_proxy" in test:
        names.update(structure_plus_pilot="tree_plus_pilot", pilot_direct="pilot_direct")
    if "pilot_only" in reference:
        names["pilot_only"] = "tree_pilot_only"
    for old, new in names.items():
        output[new] = [float(value) if value != "" else np.nan for value in reference[old]]
    columns = ["gnn"] + list(names.values())
    common = output[np.isfinite(output[columns]).all(axis=1)]
    all_graph = output[output.graph_valid]
    if len(common) < 3:
        raise ValueError("Too few matched test predictions")
    comparators = ["lomap", "tree_structure"]
    if "pilot_proxy" in test:
        comparators.extend(["tree_plus_pilot", "pilot_direct"])
    if "pilot_only" in reference:
        comparators.append("tree_pilot_only")
    def scores(frame, cols):
        return {c: dict(metrics(frame.target.to_numpy(), frame[c].to_numpy()),
                        mse=float(np.mean((frame.target - frame[c])**2)) if c != "lomap" else None)
                for c in cols}
    result = {
        "architecture": architecture, "mode": "structure_plus_pilot" if pilot else "structure",
        "test_rows": len(output), "graph_valid_rows": len(all_graph), "common_rows": len(common),
        "common_metrics": scores(common, columns),
        "all_graph_valid_metrics": scores(all_graph, [c for c in columns if c != "lomap"]),
        "paired_bootstrap": {c: bootstrap(common, ["gnn", c], c) for c in comparators},
        "per_source_common": {source: {"rows": len(part), "metrics": scores(part, columns)}
                              for source, part in common.groupby("series_name")},
        "test_reference_groups": output.reference_group.nunique(),
        "test_variable_groups": output.variable_group.nunique(),
    }
    if "pilot_only" in reference:
        result["paired_bootstrap_all_graph_valid"] = {
            c: bootstrap(all_graph, ["gnn", c], c)
            for c in ("tree_pilot_only", "tree_plus_pilot", "pilot_direct")}
    return output, result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark_dir", type=Path, required=True)
    p.add_argument("--parent_benchmark_dir", type=Path,
                   help="Required for campaign-pilot tasks to verify unchanged splits and reuse parent graphs")
    p.add_argument("--reference_dir", type=Path, required=True, help="Completed tree/LOMAP result directory")
    p.add_argument("--graph_cache", type=Path, required=True)
    p.add_argument("--architecture", choices=["hypersphere", "hybrid"], required=True)
    p.add_argument("--task", choices=["edge", "pilot_complex", "pilot_solvent"], required=True)
    p.add_argument("--mode", choices=["structure", "structure_plus_pilot"], default="structure")
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--phase", choices=["train", "evaluate", "all"], default="all")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    if args.threads < 1 or (args.task == "edge" and args.mode != "structure"):
        p.error("Use positive threads; pooled edge task is structure-only")
    torch.set_num_threads(args.threads)
    benchmark_path = args.benchmark_dir / "benchmark.json"
    benchmark_hash = file_hash(benchmark_path)
    benchmark = json.loads(benchmark_path.read_text())
    if benchmark["version"] != 2 or benchmark["rdkit"] != rdBase.rdkitVersion:
        raise ValueError("Requires frozen schema 2 and matching RDKit")
    if not benchmark["report"]["task_status"][args.task]["runnable"]:
        print(f"SKIPPED unsupported task: {benchmark['report']['task_status'][args.task]}")
        return
    table = args.benchmark_dir / (args.task + ".csv")
    if file_hash(table) != benchmark["tables"][table.name]:
        raise ValueError("Benchmark table checksum mismatch")
    frame = pd.read_csv(table, keep_default_na=False, float_precision="round_trip")
    if not frame.edge_id.is_unique or not np.isfinite(frame.target).all() or not frame.target.gt(0).all():
        raise ValueError("Invalid benchmark identities or targets")
    pilot = args.mode == "structure_plus_pilot"
    if pilot and (not np.isfinite(frame.pilot_proxy).all() or not frame.pilot_proxy.gt(0).all()):
        raise ValueError("Invalid pilot values")
    reference_path = args.reference_dir / (args.task + "_predictions.csv")
    reference_request_path = args.reference_dir / "request.json"
    reference_request = json.loads(reference_request_path.read_text())
    if (reference_request["benchmark_sha256"] != benchmark_hash
            or reference_request["lomap_version"] != LOMAP_SCORING_VERSION):
        raise ValueError("Tree/LOMAP reference belongs to another benchmark or scorer")
    completed_reference = json.loads((args.reference_dir / "results.json").read_text())
    if completed_reference["setting"] != benchmark["setting"] or args.task not in completed_reference:
        raise ValueError("Reference task is not complete")
    graph_benchmark_hash = benchmark_hash
    if "parent_benchmark_sha256" in benchmark:
        from prepare_fah_campaign_pilot import PROTOCOL, verify_parent_subset
        if (benchmark.get("protocol") != PROTOCOL or args.parent_benchmark_dir is None
                or reference_request.get("protocol") != PROTOCOL):
            raise ValueError("Campaign-pilot runs require their parent benchmark and matching pilot baselines")
        graph_benchmark_hash = verify_parent_subset(
            frame, args.parent_benchmark_dir, benchmark["parent_benchmark_sha256"])
    graphs, cache_hash = load_graph_cache(args.graph_cache, frame, args.architecture, graph_benchmark_hash)
    valid = graph_validity(frame, graphs, args.architecture)
    coverage = {}
    for split in ("train", "val", "test"):
        part = frame[frame.split.eq(split) & valid]
        coverage[split] = {"total": int(frame.split.eq(split).sum()), "graph_valid": len(part),
                           "variable_groups": part.variable_group.nunique()}
        if len(part) < 100 or part.variable_group.nunique() < 5:
            raise ValueError(f"Insufficient graph-valid coverage: {split}, {coverage[split]}")
    base = Path(__file__).parent
    request = {
        "version": 1, "benchmark_sha256": benchmark_hash, "setting": benchmark["setting"],
        "graph_parent_benchmark_sha256": graph_benchmark_hash,
        "task": args.task, "architecture": args.architecture, "mode": args.mode, "config": CONFIG,
        "graph_cache_manifest_sha256": cache_hash,
        "reference_request_sha256": file_hash(reference_request_path),
        "reference_predictions_sha256": file_hash(reference_path),
        "reference_results_sha256": file_hash(args.reference_dir / "results.json"),
        "code": {name: file_hash(base / name) for name in (
            "run_fah_gnn.py", "fah_gnn_models.py", "prepare_fah_gnn.py",
            "run_fah_benchmark.py",
            "rbfe_artifacts.py")},
        "packages": {name: version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "rdkit")},
        "torch": str(torch.__version__),
    }
    # Record actual imported graph sources, including cluster-local module names.
    request["code"][Path(sphere.__file__).name] = file_hash(Path(sphere.__file__))
    request["code"][Path(hybrid.__file__).name] = file_hash(Path(hybrid.__file__))
    if "parent_benchmark_sha256" in benchmark:
        request["code"]["prepare_fah_campaign_pilot.py"] = file_hash(base / "prepare_fah_campaign_pilot.py")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    request_path = args.output_dir / "request.json"
    if request_path.exists():
        if json.loads(request_path.read_text()) != request:
            raise ValueError("Run identity changed; do not overwrite a frozen run")
    elif any(args.output_dir.iterdir()):
        raise ValueError("Unmanaged run directory")
    else:
        atomic_json(request_path, request)
    if (args.output_dir / "results.json").exists():
        print("Completed run exists; no retraining or reevaluation.")
        return
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; submit to a GPU node")
    torch.use_deterministic_algorithms(True, warn_only=True)
    train = frame[frame.split.eq("train") & valid].reset_index(drop=True)
    val = frame[frame.split.eq("val") & valid].reset_index(drop=True)
    identity = request_hash(request)
    if args.phase in ("train", "all"):
        state = train_model(train, val, graphs, args.architecture, pilot,
                            args.output_dir, identity, device)
        if args.phase == "train":
            return
    else:
        state = torch.load(args.output_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    if state["identity"] != identity or not state["finished"] or state["best_model"] is None:
        raise ValueError("Evaluation requires the completed selected checkpoint")
    model = make_model(args.architecture, pilot, CONFIG, device)
    model.load_state_dict(state["best_model"])
    test = frame[frame.split.eq("test")].reset_index(drop=True)
    test_valid = graph_validity(test, graphs, args.architecture)
    log_prediction = predict_log(model, test[test_valid], graphs, args.architecture, pilot,
                                 state["statistics"], device, CONFIG["batch_size"])
    with np.errstate(over="raise", invalid="raise"):
        prediction = np.exp(log_prediction)
    reference = pd.read_csv(reference_path, keep_default_na=False, float_precision="round_trip")
    output, result = evaluate(test, test_valid, prediction, reference, args.architecture, pilot)
    result.update(setting=benchmark["setting"], coverage=coverage,
                  best_epoch=state["best_epoch"] + 1, best_validation_spearman=state["best_spearman"],
                  target_statistics=state["statistics"], request_sha256=identity,
                  checkpoint_sha256=file_hash(args.output_dir / "checkpoint.pt"))
    temporary = args.output_dir / "predictions.partial.csv"
    output.to_csv(temporary, index=False)
    temporary.replace(args.output_dir / "predictions.csv")
    atomic_json(args.output_dir / "results.json", result)
    print(f"Completed: {args.output_dir / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
