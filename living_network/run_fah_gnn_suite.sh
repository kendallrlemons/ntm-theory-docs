#!/usr/bin/env bash
set -euo pipefail
architecture="${1:?Usage: bash run_fah_gnn_suite.sh hypersphere|hybrid [known_reference|strict_network|both]}"
setting_choice="${2:-both}"
case "$architecture" in hypersphere|hybrid) ;; *) echo 'Unknown architecture' >&2; exit 2;; esac
case "$setting_choice" in known_reference|strict_network|both) ;; *) echo 'Unknown setting' >&2; exit 2;; esac
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FAH_ROOT="${FAH_ROOT:-/data1/choderaj/lemonsk/FEC_Sandbox}"
FAH_PYTHON="${FAH_PYTHON:-python}"
for setting in known_reference strict_network; do
  if [[ "$setting_choice" != both && "$setting_choice" != "$setting" ]]; then continue; fi
  for task in edge pilot_complex pilot_solvent; do
    if [[ "$setting" == strict_network && "$task" != edge ]]; then continue; fi
    for mode in structure structure_plus_pilot; do
      if [[ "$task" == edge && "$mode" == structure_plus_pilot ]]; then continue; fi
      "$FAH_PYTHON" "$script_dir/run_fah_gnn.py" \
        --benchmark_dir "$FAH_ROOT/fah_benchmark_v2/$setting" \
        --reference_dir "$FAH_ROOT/fah_results_${setting}_v2" \
        --graph_cache "$FAH_ROOT/fah_gnn_graphs_v1/$architecture" \
        --architecture "$architecture" --task "$task" --mode "$mode" \
        --output_dir "$FAH_ROOT/fah_gnn_results_v1/$setting/$architecture/$task/$mode" \
        --phase "${FAH_PHASE:-all}" --device "${FAH_DEVICE:-cuda}" --threads "${FAH_THREADS:-4}"
    done
  done
done
