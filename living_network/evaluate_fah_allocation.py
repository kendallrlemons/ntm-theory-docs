"""Assumption-based allocation sensitivity study using frozen campaign predictions."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROTOCOL = {
    "version": "campaign-production-allocation-v1",
    "objective": "mean predicted production-stage variance, evaluated as mean(s_gen1**2 / n)",
    "pilot_cost": "sum generation-0 raw work-pair counts; one proxy cost unit each",
    "production_model": "continuous retained/effective effort; constant cost per effort unit",
    "pilot_reuse": "pilot samples are charged but excluded from production estimate",
    "budget_multiples_of_observed_pilot_cost": [1.25, 1.5, 2., 3., 5., 10.],
    "production_unit_costs": [1., 2., 4.],
    "minimum_production_efforts": [1., 30.],
    "primary": {"production_unit_cost": 1., "minimum_production_effort": 1.},
    "allocation": "n_i = max(minimum, lambda * predicted_spread_i), sum(n) = remaining effort",
    "lomap_policy": "max(stored difficulty=1-similarity, 1e-6) as relative spread weights",
    "inference": "descriptive fixed-campaign sensitivity study; no sampling confidence intervals",
}
PILOT_POLICIES = {
    "uniform_after_pilot", "pilot_direct", "pilot_only",
    "trees_plus_pilot", "hypersphere_plus_pilot", "oracle_after_pilot",
}


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def allocate(weights, effort, minimum):
    """Optimal continuous allocation for surrogate objective sum(weights**2/n)."""
    weights = np.asarray(weights, dtype=float)
    if (weights.ndim != 1 or not len(weights) or not np.isfinite(weights).all()
            or (weights <= 0).any() or not np.isfinite(effort)
            or not np.isfinite(minimum) or minimum <= 0):
        raise ValueError("Positive finite weights and minimum, and finite effort required")
    lower = len(weights) * minimum
    if effort < lower - 1e-10 * max(1., lower):
        return None
    if effort <= lower:
        return np.full(len(weights), minimum)
    scaled = weights / weights.max()
    if (scaled == 0).any():
        raise ValueError("Weight dynamic range underflow")
    result = np.full(len(weights), minimum)
    free = np.ones(len(weights), dtype=bool)
    while free.any():
        available = effort - minimum * (~free).sum()
        proposed = available * scaled[free] / scaled[free].sum()
        below = proposed < minimum
        if not below.any():
            result[free] = proposed
            break
        positions = np.flatnonzero(free)
        free[positions[below]] = False
    if not np.isclose(result.sum(), effort, rtol=1e-10, atol=1e-8):
        raise AssertionError("Effort conservation failure")
    return result


def evaluate_policy(reference, weights, budget, pilot_cost, unit_cost, minimum):
    if unit_cost <= 0 or pilot_cost < 0 or budget <= 0:
        raise ValueError("Invalid budget or costs")
    n = allocate(weights, (budget - pilot_cost) / unit_cost, minimum)
    if n is None:
        return {"status": "infeasible", "pilot_cost_charged": float(pilot_cost)}
    consumed = pilot_cost + unit_cost * n.sum()
    if not np.isclose(consumed, budget, rtol=1e-10):
        raise AssertionError("Total budget conservation failure")
    variance = float(np.mean(np.asarray(reference)**2 / n))
    if not np.isfinite(variance):
        raise ValueError("Nonfinite reference variance")
    return {
        "status": "ok", "pilot_cost_charged": float(pilot_cost),
        "production_effort": float(n.sum()), "total_cost": float(consumed),
        "mean_variance_proxy": variance, "min_effort": float(n.min()),
        "max_effort": float(n.max()), "fraction_at_floor": float(np.mean(n <= minimum * (1 + 1e-10))),
    }


def floor_free_shape(reference, weights):
    """Risk ratio to uniform at equal production effort, ignoring effort floors."""
    reference, weights = np.asarray(reference, float), np.asarray(weights, float)
    return float(weights.sum() * np.sum(reference**2 / weights)
                 / (len(reference) * np.sum(reference**2)))


def load_phase(root, review, task):
    baseline_path = root / "trees" / f"{task}_predictions.csv"
    identity = review["tasks"][task]
    if sha(baseline_path) != identity["predictions_sha256"]:
        raise ValueError("Baseline CSV differs from verified review")
    base = pd.read_csv(baseline_path, float_precision="round_trip")
    if not base.edge_id.is_unique or not base.split.eq("test").all():
        raise ValueError("Invalid test identities")
    out = base[["edge_id", "source_sha256", "series_name", "variable_group", "reference_group",
                "target", "early_raw_work_pair_count", "early_num_work_pairs", "lomap"]].copy()
    out["trees_structure"] = base.structure
    out["pilot_only"] = base.pilot_only
    out["trees_plus_pilot"] = base.structure_plus_pilot
    out["pilot_direct"] = base.pilot_direct
    hashes = {str(baseline_path): sha(baseline_path)}
    for mode, column in (("structure", "hypersphere_structure"),
                         ("structure_plus_pilot", "hypersphere_plus_pilot")):
        path = root / "hypersphere" / task / mode / "predictions.csv"
        if sha(path) != identity["gnns"][mode]["predictions_sha256"]:
            raise ValueError("GNN CSV differs from verified review")
        g = pd.read_csv(path, float_precision="round_trip", keep_default_na=False)
        if not g.edge_id.is_unique or set(g.edge_id) != set(base.edge_id):
            raise ValueError("GNN cohort differs")
        g = g.set_index("edge_id").loc[base.edge_id]
        for col in ("smi_a", "smi_b", "source_sha256"):
            if not np.array_equal(base[col], g[col]):
                raise ValueError(f"Identity mismatch: {col}")
        if not np.allclose(base.target, g.target, rtol=1e-12, atol=1e-12) or not g.graph_valid.all():
            raise ValueError("Target mismatch or invalid graphs")
        out[column] = g.gnn.to_numpy()
        hashes[str(path)] = sha(path)
    required = ["target", "early_raw_work_pair_count", "early_num_work_pairs",
                "trees_structure", "pilot_only", "trees_plus_pilot", "pilot_direct",
                "hypersphere_structure", "hypersphere_plus_pilot"]
    if not np.isfinite(out[required]).all().all() or not out[required].gt(0).all().all():
        raise ValueError("Nonpositive/nonfinite inputs; do not silently trim cohorts")
    return out, hashes


def study(frame, task, cohort, include_lomap):
    reference = frame.target.to_numpy()
    cost = float(frame.early_raw_work_pair_count.sum())
    weights = {
        "uniform": np.ones(len(frame)),
        "trees_structure": frame.trees_structure.to_numpy(),
        "hypersphere_structure": frame.hypersphere_structure.to_numpy(),
        "uniform_after_pilot": np.ones(len(frame)),
        "pilot_direct": frame.pilot_direct.to_numpy(),
        "pilot_only": frame.pilot_only.to_numpy(),
        "trees_plus_pilot": frame.trees_plus_pilot.to_numpy(),
        "hypersphere_plus_pilot": frame.hypersphere_plus_pilot.to_numpy(),
        "oracle_no_pilot": reference,
        "oracle_after_pilot": reference,
    }
    if include_lomap:
        if not np.isfinite(frame.lomap).all() or not frame.lomap.between(0, 1).all():
            raise ValueError("Invalid LOMAP difficulty")
        weights["lomap_fixed"] = np.maximum(frame.lomap.to_numpy(), 1e-6)
    rows, shape = [], []
    for policy, w in weights.items():
        factor = floor_free_shape(reference, w)
        shape.append({
            "task": task, "cohort": cohort, "policy": policy,
            "equal_production_effort_risk_ratio": factor,
            "floor_free_break_even_B_over_pilot_cost": (
                1 / (1 - factor) if policy in PILOT_POLICIES and factor < 1 - 1e-12 else None),
            "interpretation": "analytic screen only; floors can change the crossover",
        })
    for multiple in PROTOCOL["budget_multiples_of_observed_pilot_cost"]:
        budget = multiple * cost
        for unit in PROTOCOL["production_unit_costs"]:
            for minimum in PROTOCOL["minimum_production_efforts"]:
                baseline = evaluate_policy(reference, weights["uniform"], budget, 0., unit, minimum)
                oracle = evaluate_policy(reference, reference, budget, 0., unit, minimum)
                for policy, w in weights.items():
                    charged = cost if policy in PILOT_POLICIES else 0.
                    result = evaluate_policy(reference, w, budget, charged, unit, minimum)
                    row = dict(task=task, cohort=cohort, policy=policy, rows=len(frame),
                               budget_multiple=multiple, total_budget=budget,
                               production_unit_cost=unit, minimum_effort=minimum, **result)
                    if result["status"] == "ok":
                        risk = result["mean_variance_proxy"]
                        if risk < oracle["mean_variance_proxy"] * (1 - 1e-10):
                            raise AssertionError("Policy beats no-cost oracle bound")
                        ratio = risk / baseline["mean_variance_proxy"]
                        row.update(risk_ratio_to_uniform=ratio,
                                   variance_reduction_percent=100 * (1 - ratio))
                    rows.append(row)
    return rows, shape


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results_root", required=True, type=Path)
    p.add_argument("--verified_review", required=True, type=Path)
    p.add_argument("--output_dir", required=True, type=Path)
    args = p.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Use a fresh allocation-study output directory")
    review = json.loads(args.verified_review.read_text())
    inputs, hashes, rows, shapes, coverage = {}, {}, [], [], {}
    for task in ("pilot_complex", "pilot_solvent"):
        frame, phase_hashes = load_phase(args.results_root, review, task)
        inputs[task] = frame
        hashes.update(phase_hashes)
        coverage[task] = {
            "eligible_rows": len(frame), "lomap_finite_rows": int(np.isfinite(frame.lomap).sum()),
            "sources": frame.source_sha256.nunique(), "reference_groups": frame.reference_group.nunique(),
            "pilot_raw_cost_total": float(frame.early_raw_work_pair_count.sum()),
            "pilot_retained_count_total": float(frame.early_num_work_pairs.sum()),
        }
        for cohort, part, include in (
            ("primary_all", frame, False),
            ("lomap_common", frame[np.isfinite(frame.lomap)].copy(), True),
        ):
            result, shape = study(part, task, cohort, include)
            rows.extend(result)
            shapes.extend(shape)
    # Preserve a frozen manifest and the exact inputs needed to reproduce allocations.
    args.output_dir.mkdir(parents=True)
    request = {"protocol": PROTOCOL, "script_sha256": sha(Path(__file__)),
               "verified_review_sha256": sha(args.verified_review),
               "input_prediction_hashes": hashes,
               "packages": {"numpy": np.__version__, "pandas": pd.__version__},
               "coverage": coverage}
    (args.output_dir / "request.json").write_text(json.dumps(request, indent=2) + "\n")
    for task, frame in inputs.items():
        frame.to_csv(args.output_dir / f"{task}_policy_inputs.csv", index=False)
    result = pd.DataFrame(rows)
    result.to_csv(args.output_dir / "allocation_results.csv", index=False)
    (args.output_dir / "floor_free_diagnostics.json").write_text(json.dumps(shapes, indent=2) + "\n")
    primary = result[(result.cohort == "primary_all") & result.production_unit_cost.eq(1)
                     & result.minimum_effort.eq(1)]
    primary.to_csv(args.output_dir / "primary_results.csv", index=False)
    print(primary.pivot(index=["task", "policy"], columns="budget_multiple",
                        values="risk_ratio_to_uniform").round(4).to_string())
    print(f"Saved assumption-based study: {args.output_dir}")


if __name__ == "__main__":
    main()
