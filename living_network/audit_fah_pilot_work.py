"""Targeted source-work audit. Does not filter works, replace labels, or fit models."""
import argparse
import csv
import json
import math
from pathlib import Path
import shutil
import statistics
import tempfile

import ijson

from fah_edge_dataset import digest


def log_sigmoid(x):
    return -max(0., -x) - math.log1p(math.exp(-abs(x)))


def log_mean_exp(values):
    peak = max(values)
    return peak + math.log(math.fsum(math.exp(x - peak) for x in values)) - math.log(len(values))


def bar_log_balance(forward, reverse, delta):
    """Sign-equivalent BAR balance, using complementary tails near saturation."""
    left = log_mean_exp([log_sigmoid(delta - x) for x in forward])
    right = log_mean_exp([log_sigmoid(-delta - x) for x in reverse])
    if left > -math.log(2) and right > -math.log(2):
        # If both means round to one, their logs cannot resolve equality.
        # Compare (1 - reverse acceptance) to (1 - forward acceptance).
        return (log_mean_exp([log_sigmoid(delta + x) for x in reverse])
                - log_mean_exp([log_sigmoid(x - delta) for x in forward]))
    return left - right


def diagnostic_bar_root(forward, reverse):
    """Stable bisection of the monotone equal-count equation; no error estimate."""
    if not forward or len(forward) != len(reverse):
        raise ValueError("Requires nonempty paired forward/reverse works")
    if not all(math.isfinite(x) for x in forward + reverse):
        raise ValueError("Nonfinite works: do not silently filter the diagnostic")
    left = min(min(forward), min(-x for x in reverse)) - 50.
    right = max(max(forward), max(-x for x in reverse)) + 50.
    if not math.isfinite(left + right):
        raise ValueError("Work magnitude exceeds diagnostic range")
    if bar_log_balance(forward, reverse, left) >= 0 or bar_log_balance(forward, reverse, right) <= 0:
        raise ValueError("Could not bracket diagnostic BAR root")
    for _ in range(200):
        mid = left + (right - left) / 2
        balance = bar_log_balance(forward, reverse, mid)
        if balance < 0:
            left = mid
        else:
            right = mid
        if right - left <= 1e-11 * max(1., abs(mid)):
            break
    root = left + (right - left) / 2
    return {"diagnostic_delta_f": root,
            "diagnostic_log_balance_at_root": bar_log_balance(forward, reverse, root),
            "diagnostic_forward_log_mean_acceptance":
                log_mean_exp([log_sigmoid(root - x) for x in forward]),
            "diagnostic_reverse_log_mean_acceptance":
                log_mean_exp([log_sigmoid(-root - x) for x in reverse]),
            "diagnostic_forward_log_mean_complement":
                log_mean_exp([log_sigmoid(x - root) for x in forward]),
            "diagnostic_reverse_log_mean_complement":
                log_mean_exp([log_sigmoid(root + x) for x in reverse])}


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def work_statistics(values, prefix):
    finite = [x for x in values if finite_number(x)]
    return {
        prefix + "_finite": len(finite), prefix + "_nonfinite": len(values) - len(finite),
        prefix + "_min": min(finite) if finite else None,
        prefix + "_max": max(finite) if finite else None,
        prefix + "_mean": statistics.mean(finite) if finite else None,
        prefix + "_sample_std": statistics.stdev(finite) if len(finite) > 1 else None,
        prefix + "_unique_finite": len(set(finite)),
    }


def summarize_generation(gen):
    works = gen["works"]
    forward, reverse = [], []
    for work in works:
        forward.append(work["forward"])
        reverse.append(work["reverse"])
    clones = [work.get("clone") for work in works]
    known_clones = [x for x in clones if x is not None]
    free = gen.get("free_energy") or {}
    estimate = free.get("delta_f") or {}
    row = {
        "generation": gen["gen"], "exported_pairs": len(works),
        "reported_retained_pairs": free.get("num_work_pairs"),
        "exported_count_matches_reported": len(works) == free.get("num_work_pairs"),
        "reported_delta_f": estimate.get("point"), "reported_stderr": estimate.get("stderr"),
        "reported_bar_overlap": free.get("bar_overlap"),
        "unique_clones": len(set(known_clones)), "missing_clone_ids": clones.count(None),
        "duplicate_clone_records": len(known_clones) - len(set(known_clones)),
        **work_statistics(forward, "forward"), **work_statistics(reverse, "reverse"),
    }
    # Only finite, complete exported arrays enter this diagnostic; no new outlier filter.
    eligible = len(works) > 0 and all(finite_number(x) for x in forward + reverse)
    row.update(diagnostic_status="not_computed_nonfinite_or_empty",
               diagnostic_delta_f=None, diagnostic_log_balance_at_root=None,
               diagnostic_forward_log_mean_acceptance=None,
               diagnostic_reverse_log_mean_acceptance=None,
               diagnostic_forward_log_mean_complement=None,
               diagnostic_reverse_log_mean_complement=None,
               diagnostic_log_balance_at_reported_delta=None,
               diagnostic_minus_reported_delta=None)
    if eligible:
        row.update(diagnostic_bar_root(forward, reverse))
        row["diagnostic_status"] = "computed_on_exported_works"
        if finite_number(estimate.get("point")):
            row["diagnostic_log_balance_at_reported_delta"] = bar_log_balance(
                forward, reverse, estimate["point"])
            row["diagnostic_minus_reported_delta"] = row["diagnostic_delta_f"] - estimate["point"]
    return row


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No records for {path.name}")
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit(analysis, specification, output):
    if output.exists():
        raise FileExistsError("Use a fresh audit output directory")
    if analysis.stat().st_size != specification["source_bytes"] or digest(analysis) != specification["source_sha256"]:
        raise ValueError("Source snapshot size/hash differs from the frozen audit specification")
    cases, controls = set(specification["case_indices"]), set(specification["control_indices"])
    if cases & controls or len(cases) != len(specification["case_indices"]) or len(controls) != len(specification["control_indices"]):
        raise ValueError("Repeated or overlapping case/control indices")
    selected, diagnostics, work_rows, reuse = [], [], [], []
    found = set()
    with open(analysis, "rb") as handle:
        total = 0
        for index, entry in enumerate(ijson.items(handle, "series.transformations.item", use_float=True)):
            total += 1
            if index not in cases | controls:
                continue
            found.add(index)
            identity = {
                "edge_id": f"{specification['source_sha256']}:{index}",
                "source_transformation_index": index, "run_id": entry["transformation"]["run_id"],
                "cohort": "case" if index in cases else "control",
            }
            selected.append(dict(identity, transformation_record=entry))
            for phase in specification["phases"]:
                generations = entry[phase]["gens"]
                seen = set()
                clone_sets = {}
                for gen in generations:
                    if gen["gen"] in seen:
                        raise ValueError(f"Duplicate generation: {identity['edge_id']}/{phase}")
                    seen.add(gen["gen"])
                    phase_id = dict(identity, phase=phase)
                    diagnostics.append(dict(phase_id, **summarize_generation(gen)))
                    clone_sets[gen["gen"]] = {w["clone"] for w in gen["works"] if w.get("clone") is not None}
                    for work_index, work in enumerate(gen["works"]):
                        work_rows.append(dict(phase_id, generation=gen["gen"], work_index=work_index,
                                              clone=work.get("clone"), forward=work["forward"], reverse=work["reverse"]))
                early, late = clone_sets.get(0, set()), clone_sets.get(1, set())
                reuse.append(dict(identity, phase=phase, gen0_exported=0 in seen, gen1_exported=1 in seen,
                                  gen0_unique_clones=len(early), gen1_unique_clones=len(late),
                                  shared_clone_ids=len(early & late)))
    if total != specification["expected_transformations"] or found != cases | controls:
        raise ValueError("Source transformation count or selected indices differ")
    if digest(analysis) != specification["source_sha256"]:
        raise ValueError("Source changed during audit")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=output.parent, prefix=".pilot-work-audit-"))
    try:
        write_csv(staging / "generation_diagnostics.csv", diagnostics)
        write_csv(staging / "work_pairs.csv", work_rows)
        write_csv(staging / "clone_reuse.csv", reuse)
        (staging / "selected_transformations.json").write_text(json.dumps(selected, indent=2, allow_nan=False) + "\n")
        report = {
            "source_path": str(analysis.resolve()), "source_sha256": specification["source_sha256"],
            "specification": specification, "script_sha256": digest(Path(__file__)),
            "selected_transformations": len(selected), "generation_records": len(diagnostics),
            "exported_work_pairs": len(work_rows),
            "notes": [
                "Exported works may already be filtered by the historical analysis; these are not necessarily unfiltered trajectories.",
                "No historical labels are recomputed or replaced. No uncertainty is estimated by the diagnostic solver.",
                "Diagnostic assumes standard dimensionless forward/reverse work convention and equal counts.",
                "Check diagnostic agreement on controls before interpreting disagreement on cases.",
                "Reported-count equality does not prove identical filtering or identify the historical software version.",
                "Shared clone IDs establish reuse of identifiers, not independent replicate evidence.",
            ],
            "outputs": {p.name: digest(p) for p in staging.iterdir()},
        }
        (staging / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--analysis", required=True, type=Path)
    p.add_argument("--specification", type=Path, default=Path(__file__).with_name("fah_low_overlap_audit_cases.json"))
    p.add_argument("--output_dir", required=True, type=Path)
    args = p.parse_args()
    report = audit(args.analysis, json.loads(args.specification.read_text()), args.output_dir)
    print(json.dumps({k: report[k] for k in ("selected_transformations", "generation_records", "exported_work_pairs")}, indent=2))
    print(f"Audit complete: {args.output_dir}")


if __name__ == "__main__":
    main()
