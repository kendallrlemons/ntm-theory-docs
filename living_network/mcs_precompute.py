"""
mcs_precompute.py
=================
Standalone script to precompute hybrid-topology molecular graphs from a
compound-pair CSV and save them to disk for use by hybrid_topo_rbfe.py.

This separates the slow MCS computation (rdFMCS.FindMCS per pair) from
model training, so it can be parallelized independently (e.g. SLURM array)
and reused across many training runs without recomputing.

Output (saved to --output_dir):
    cleaned_split.pkl          — cleaned DataFrame with split column
    graphs_train.pt            — list of graph dicts (or None) for train rows
    graphs_val.pt              — list of graph dicts (or None) for val rows
    graphs_test.pt             — list of graph dicts (or None) for test rows

    If --num_chunks > 1:
    graphs_{split}_chunk{K}_of_{N}.pt  — partial files, merged at the end

These files are directly loadable by hybrid_topo_rbfe.py's _load_graphs().

Usage — single process (small dataset or test run):
    python mcs_precompute.py --input data.csv --output_dir ./graphs

Usage — parallel across SLURM array (recommended for full dataset):
    # In SLURM script with --array=0-15:
    python mcs_precompute.py \\
        --input data.csv \\
        --output_dir ./graphs \\
        --num_chunks 16 \\
        --chunk_id $SLURM_ARRAY_TASK_ID

    # After all array tasks finish, merge:
    python mcs_precompute.py --merge_only --output_dir ./graphs --num_chunks 16
"""

from __future__ import annotations

import argparse
import os
import pickle
import signal
import sys
import threading
import time
from multiprocessing import cpu_count, get_context
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from rdkit import Chem, RDLogger
from rdkit.Chem import rdFMCS

RDLogger.DisableLog("rdApp.*")

# =============================================================================
# Column names (must match hybrid_topo_rbfe.py)
# =============================================================================

SE_COLUMN       = "Difference in first_pass_free_energy_stderr"
SMILES_A_COLUMN = "Compound Smiles 1"
SMILES_B_COLUMN = "Compound Smiles 2"

# =============================================================================
# Atom / bond vocabularies (must match hybrid_topo_rbfe.py exactly)
# =============================================================================

ATOM_VOCAB = {
    "atomic_num":    list(range(1, 120)),
    "degree":        list(range(0, 7)),
    "formal_charge": [-2, -1, 0, 1, 2],
    "hybridization": [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ],
    "num_hs": list(range(0, 5)),
}

BOND_VOCAB = {
    "bond_type": [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ],
}


def _one_hot(v, vocab):
    x = [0] * (len(vocab) + 1)
    if v in vocab:
        x[vocab.index(v)] = 1
    else:
        x[-1] = 1
    return x


def _base_atom_features(atom) -> List[float]:
    return (
        _one_hot(atom.GetAtomicNum(),       ATOM_VOCAB["atomic_num"])
        + _one_hot(atom.GetDegree(),         ATOM_VOCAB["degree"])
        + _one_hot(atom.GetFormalCharge(),   ATOM_VOCAB["formal_charge"])
        + _one_hot(atom.GetHybridization(),  ATOM_VOCAB["hybridization"])
        + _one_hot(atom.GetTotalNumHs(),     ATOM_VOCAB["num_hs"])
        + [int(atom.GetIsAromatic()),
           int(atom.IsInRing()),
           float(atom.GetNumRadicalElectrons())]
    )


def _bond_features(bond) -> List[float]:
    return (
        _one_hot(bond.GetBondType(), BOND_VOCAB["bond_type"])
        + [int(bond.GetIsConjugated()),
           int(bond.IsInRing()),
           int(bond.GetStereo() != Chem.rdchem.BondStereo.STEREONONE)]
    )


ENDPOINT_DIM = 3
_BASE_DIM = (
    len(ATOM_VOCAB["atomic_num"]) + 1
    + len(ATOM_VOCAB["degree"]) + 1
    + len(ATOM_VOCAB["formal_charge"]) + 1
    + len(ATOM_VOCAB["hybridization"]) + 1
    + len(ATOM_VOCAB["num_hs"]) + 1
    + 3
)
ATOM_DIM = _BASE_DIM + ENDPOINT_DIM   # 152
BOND_DIM = len(BOND_VOCAB["bond_type"]) + 1 + 3  # 8

# =============================================================================
# Hybrid graph builder  (identical to hybrid_topo_rbfe.py)
# =============================================================================

def build_hybrid_graph(smi_a: str, smi_b: str,
                       mcs_timeout: int = 2) -> Optional[Dict]:
    mol_a = Chem.MolFromSmiles(smi_a)
    mol_b = Chem.MolFromSmiles(smi_b)
    if mol_a is None or mol_b is None:
        return None

    try:
        res = rdFMCS.FindMCS(
            [mol_a, mol_b],
            atomCompare=rdFMCS.AtomCompare.CompareElements,
            bondCompare=rdFMCS.BondCompare.CompareOrder,
            completeRingsOnly=False,
            timeout=mcs_timeout,
        )
        mcs_mol = Chem.MolFromSmarts(res.smartsString) if res.numAtoms > 0 else None
    except Exception:
        mcs_mol = None

    if mcs_mol is not None and mcs_mol.GetNumAtoms() > 0:
        match_a = mol_a.GetSubstructMatch(mcs_mol)
        match_b = mol_b.GetSubstructMatch(mcs_mol)
    else:
        match_a = ()
        match_b = ()

    core_a_idx = set(match_a)
    core_b_idx = set(match_b)
    uniq_a_idx = [i for i in range(mol_a.GetNumAtoms()) if i not in core_a_idx]
    uniq_b_idx = [i for i in range(mol_b.GetNumAtoms()) if i not in core_b_idx]

    n_core     = len(match_a)
    n_a_unique = len(uniq_a_idx)
    n_b_unique = len(uniq_b_idx)

    a_to_merged: Dict[int, int] = {}
    for merged_idx, a_idx in enumerate(match_a):
        a_to_merged[a_idx] = merged_idx
    for local_idx, a_idx in enumerate(uniq_a_idx):
        a_to_merged[a_idx] = n_core + local_idx

    b_to_merged: Dict[int, int] = {}
    for merged_idx, b_idx in enumerate(match_b):
        b_to_merged[b_idx] = merged_idx
    for local_idx, b_idx in enumerate(uniq_b_idx):
        b_to_merged[b_idx] = n_core + n_a_unique + local_idx

    total_nodes = n_core + n_a_unique + n_b_unique
    if total_nodes == 0:
        return None

    node_feats = []
    for a_idx in match_a:
        node_feats.append(_base_atom_features(mol_a.GetAtomWithIdx(a_idx)) + [1, 0, 0])
    for a_idx in uniq_a_idx:
        node_feats.append(_base_atom_features(mol_a.GetAtomWithIdx(a_idx)) + [0, 1, 0])
    for b_idx in uniq_b_idx:
        node_feats.append(_base_atom_features(mol_b.GetAtomWithIdx(b_idx)) + [0, 0, 1])

    edge_src, edge_dst, edge_feats = [], [], []

    def _add_bond(src, dst, bond):
        bf = _bond_features(bond)
        edge_src.extend([src, dst])
        edge_dst.extend([dst, src])
        edge_feats.extend([bf, bf])

    for bond in mol_a.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        _add_bond(a_to_merged[i], a_to_merged[j], bond)

    for bond in mol_b.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i not in core_b_idx or j not in core_b_idx:
            _add_bond(b_to_merged[i], b_to_merged[j], bond)

    if not node_feats:
        return None

    # Return plain numpy arrays (not torch tensors) so inter-process results
    # pickle normally and never trigger torch's shared-memory IPC reducers,
    # which can exhaust /dev/shm or /tmp when many workers return many small
    # tensors concurrently. Converted to tensors once, in the main process,
    # after collection (see precompute_all).
    return {
        "node_feats": np.asarray(node_feats, dtype=np.float32),
        "edge_index": np.asarray([edge_src, edge_dst], dtype=np.int64)
                      if edge_src else np.zeros((2, 0), dtype=np.int64),
        "edge_feats": np.asarray(edge_feats, dtype=np.float32)
                      if edge_feats else np.zeros((0, BOND_DIM), dtype=np.float32),
        "n_core":     n_core,
        "n_a_unique": n_a_unique,
        "n_b_unique": n_b_unique,
    }


# =============================================================================
# Worker function  (top-level required for multiprocessing pickling)
# =============================================================================

_MCS_TIMEOUT = 2  # module-level so worker can access it without args


def _hard_timeout_handler(signum, frame):
    raise TimeoutError("rdFMCS hard timeout (SIGALRM)")


def _worker(args_tuple: Tuple[int, str, str]) -> Tuple[int, Optional[Dict]]:
    idx, smi_a, smi_b = args_tuple
    t_pair = time.time()
    signal.signal(signal.SIGALRM, _hard_timeout_handler)
    signal.alarm(_MCS_TIMEOUT + 2)  # hard kill 2s after the soft rdFMCS timeout
    try:
        result = build_hybrid_graph(smi_a, smi_b, mcs_timeout=_MCS_TIMEOUT)
    except TimeoutError:
        elapsed = time.time() - t_pair
        print(f"  [HARD TIMEOUT] {elapsed:.1f}s  "
              f"A={smi_a[:50]}  B={smi_b[:50]}", flush=True)
        result = None
    except Exception as exc:
        print(f"  [WORKER ERROR] {exc}  "
              f"A={smi_a[:50]}  B={smi_b[:50]}", flush=True)
        result = None
    finally:
        signal.alarm(0)  # cancel alarm so it doesn't fire between pairs
    elapsed = time.time() - t_pair
    if elapsed > _MCS_TIMEOUT * 1.5 and result is not None:
        print(f"  [SLOW PAIR] {elapsed:.1f}s  "
              f"A={smi_a[:50]}  B={smi_b[:50]}", flush=True)
    return idx, result


def _init_worker(timeout: int):
    global _MCS_TIMEOUT
    _MCS_TIMEOUT = timeout
    torch.multiprocessing.set_sharing_strategy("file_system")


# =============================================================================
# Data loading and split
# =============================================================================

def load_and_clean(input_path: str, output_dir: str,
                   sample_size: int, seed: int) -> pd.DataFrame:
    pkl = os.path.join(output_dir, "cleaned.pkl")
    if os.path.exists(pkl):
        print(f"[data] Loading cached cleaned df from {pkl}", flush=True)
        return pd.read_pickle(pkl)

    print(f"[data] Loading {input_path} ...", flush=True)
    df = pd.read_csv(input_path, low_memory=False)
    print(f"[data] Raw rows: {len(df):,}", flush=True)

    df = df[[SMILES_A_COLUMN, SMILES_B_COLUMN, SE_COLUMN]].dropna()
    df = df.drop_duplicates(subset=[SMILES_A_COLUMN, SMILES_B_COLUMN])
    df["SE_abs"] = df[SE_COLUMN].abs()
    df = df[df["SE_abs"] > 0].reset_index(drop=True)
    print(f"[data] After cleaning: {len(df):,}", flush=True)

    if sample_size > 0 and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)
        print(f"[data] Subsampled to {sample_size:,}", flush=True)

    df.to_pickle(pkl)
    print(f"[data] Saved cleaned df ({len(df):,} rows) to {pkl}", flush=True)
    return df


# =============================================================================
# Precompute all rows (or one chunk of them)
# =============================================================================

def precompute_all(df: pd.DataFrame, output_dir: str,
                   mcs_timeout: int, num_workers: int,
                   chunk_id: Optional[int], num_chunks: Optional[int]) -> str:
    """
    Compute hybrid graphs for every row in df (the full cleaned dataset).
    If chunk_id/num_chunks are given, only process that slice and save a
    partial file; call merge_chunks afterwards to assemble graphs_all.pt.
    Returns the path of the saved .pt file.
    """
    all_pairs = list(zip(df[SMILES_A_COLUMN], df[SMILES_B_COLUMN]))

    if num_chunks is not None and num_chunks > 1:
        chunk_size = int(np.ceil(len(all_pairs) / num_chunks))
        start = chunk_id * chunk_size
        end   = min(start + chunk_size, len(all_pairs))
        pairs    = all_pairs[start:end]
        out_path = os.path.join(
            output_dir, f"graphs_all_chunk{chunk_id}_of_{num_chunks}.pt"
        )
        label = f"chunk {chunk_id}/{num_chunks} ({len(pairs):,} pairs)"
    else:
        pairs    = all_pairs
        out_path = os.path.join(output_dir, "graphs_all.pt")
        label    = f"all ({len(pairs):,} pairs)"

    # Tag each pair with its local index so imap_unordered results can be sorted
    rows    = [(i, smi_a, smi_b) for i, (smi_a, smi_b) in enumerate(pairs)]
    n_total = len(rows)

    if os.path.exists(out_path):
        print(f"[precompute] Cache hit — {out_path}", flush=True)
        return out_path

    print(f"[precompute] Computing {label} with {num_workers} workers ...", flush=True)
    t0 = time.time()

    LOG_EVERY  = 1_000
    HANG_WARN  = max(60, (mcs_timeout + 2) * 4)  # warn if idle this many seconds

    last_result_t = [time.time()]
    watchdog_stop = [False]
    result_dict   = {}          # shared: populated by main loop, read by watchdog
    pool_ref      = [None]      # holds the live Pool once created
    kill_count    = [0]

    def _watchdog():
        while not watchdog_stop[0]:
            time.sleep(15)
            idle = time.time() - last_result_t[0]
            if idle > HANG_WARN:
                completed = len(result_dict)
                print(f"  [WATCHDOG] No result in {idle:.0f}s — "
                      f"completed={completed:,}/{n_total:,}", flush=True)
                lo = completed
                hi = min(completed + num_workers, n_total)
                for j in range(lo, hi):
                    if j not in result_dict:
                        _, smi_a, smi_b = rows[j]
                        print(f"    stuck? idx={j}  A={smi_a[:60]}  B={smi_b[:60]}", flush=True)
                pool = pool_ref[0]
                if pool is not None:
                    kill_count[0] += 1
                    print(f"  [WATCHDOG] Force-killing all {num_workers} workers "
                          f"(kill #{kill_count[0]}) so Pool respawns fresh ones ...", flush=True)
                    for w in list(pool._pool):
                        if w.is_alive():
                            try:
                                os.kill(w.pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                last_result_t[0] = time.time()  # avoid re-triggering every 15s

    wd = threading.Thread(target=_watchdog, daemon=True)
    wd.start()

    if num_workers > 1:
        ctx = get_context("forkserver")  # avoids inherited RDKit thread locks
        with ctx.Pool(
            processes=num_workers,
            initializer=_init_worker,
            initargs=(mcs_timeout,),
        ) as pool:
            pool_ref[0] = pool
            t_block = time.time()
            # imap_unordered: results arrive as workers finish — one stuck pair
            # only blocks its own slot, not the other (num_workers-1) workers.
            # chunksize=1 ensures each worker handles exactly one pair at a time.
            for idx, g in pool.imap_unordered(_worker, rows, chunksize=1):
                result_dict[idx] = g
                last_result_t[0] = time.time()
                completed = len(result_dict)
                if completed % LOG_EVERY == 0:
                    elapsed   = time.time() - t0
                    block_t   = time.time() - t_block
                    rate_inst = LOG_EVERY / max(block_t, 1e-6)
                    rate_avg  = completed / elapsed
                    eta       = (n_total - completed) / max(rate_avg, 1e-6)
                    pct_ok    = sum(x is not None for x in result_dict.values()) / completed * 100
                    print(f"  {completed:,}/{n_total:,}  "
                          f"{rate_inst:.1f} pairs/s (last 1k)  "
                          f"avg {rate_avg:.1f}/s  "
                          f"ETA {eta:.0f}s  "
                          f"{pct_ok:.1f}% ok", flush=True)
                    t_block = time.time()
        graphs = [result_dict.get(i) for i in range(n_total)]
    else:
        _init_worker(mcs_timeout)
        t_block = time.time()
        for idx, smi_a, smi_b in rows:
            _, g = _worker((idx, smi_a, smi_b))
            result_dict[idx] = g
            last_result_t[0] = time.time()
            completed = len(result_dict)
            if completed % LOG_EVERY == 0:
                elapsed   = time.time() - t0
                block_t   = time.time() - t_block
                rate_inst = LOG_EVERY / max(block_t, 1e-6)
                rate_avg  = completed / elapsed
                eta       = (n_total - completed) / max(rate_avg, 1e-6)
                print(f"  {completed:,}/{n_total:,}  "
                      f"{rate_inst:.1f} pairs/s (last {LOG_EVERY:,})  "
                      f"avg {rate_avg:.1f}/s  "
                      f"ETA {eta:.0f}s", flush=True)
                t_block = time.time()
        graphs = [result_dict.get(i) for i in range(n_total)]

    watchdog_stop[0] = True

    # Convert numpy arrays -> torch tensors here in the main process (single
    # thread, no IPC involved) so the saved .pt file matches the tensor-based
    # format hybrid_topo_rbfe.py expects.
    for g in graphs:
        if g is not None:
            g["node_feats"] = torch.from_numpy(g["node_feats"])
            g["edge_index"] = torch.from_numpy(g["edge_index"])
            g["edge_feats"] = torch.from_numpy(g["edge_feats"])

    n_ok = sum(g is not None for g in graphs)
    elapsed = time.time() - t0
    print(f"[precompute] Done: {n_ok:,}/{len(graphs):,} ok  ({elapsed:.0f}s)", flush=True)

    cores  = [g["n_core"]     for g in graphs if g is not None]
    uniqs  = [g["n_a_unique"] + g["n_b_unique"] for g in graphs if g is not None]
    if cores:
        print(f"[precompute] MCS core:   mean={np.mean(cores):.1f}  "
              f"min={min(cores)}  max={max(cores)}", flush=True)
        print(f"[precompute] Uniq atoms: mean={np.mean(uniqs):.1f}  "
              f"min={min(uniqs)}  max={max(uniqs)}", flush=True)

    torch.save(graphs, out_path)
    gb = os.path.getsize(out_path) / 1e9
    print(f"[precompute] Saved → {out_path} ({gb:.2f} GB)", flush=True)
    return out_path


# =============================================================================
# Merge chunks into graphs_all.pt
# =============================================================================

def merge_chunks(output_dir: str, num_chunks: int):
    out_path = os.path.join(output_dir, "graphs_all.pt")
    if os.path.exists(out_path):
        print(f"[merge] Already exists: {out_path}", flush=True)
        return

    print(f"[merge] Merging {num_chunks} chunks into graphs_all.pt ...", flush=True)
    merged = []
    for k in range(num_chunks):
        chunk_path = os.path.join(
            output_dir, f"graphs_all_chunk{k}_of_{num_chunks}.pt"
        )
        if not os.path.exists(chunk_path):
            raise FileNotFoundError(
                f"Missing chunk file: {chunk_path}\n"
                f"Make sure all {num_chunks} array tasks completed successfully."
            )
        chunk = torch.load(chunk_path, weights_only=False)
        merged.extend(chunk)
        print(f"  chunk {k}: {len(chunk):,} graphs (running total: {len(merged):,})", flush=True)

    torch.save(merged, out_path)
    gb = os.path.getsize(out_path) / 1e9
    print(f"[merge] Saved → {out_path} ({gb:.2f} GB, {len(merged):,} graphs)", flush=True)


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Precompute MCS hybrid graphs from a compound-pair CSV."
    )
    p.add_argument("--input",        default="/Users/lemonsk/Downloads/compound_smiles_stderr_differences.csv",
                   help="Path to compound-pair CSV")
    p.add_argument("--output_dir",   required=True,
                   help="Directory to save graphs_all.pt and cleaned.pkl")
    p.add_argument("--sample_size",  type=int, default=0,
                   help="Rows to keep after cleaning (0 = keep all)")
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--mcs_timeout",  type=int, default=2,
                   help="MCS search timeout per pair in seconds (default 2)")
    p.add_argument("--num_workers",  type=int, default=max(1, cpu_count() - 1),
                   help="Parallel worker processes (default: nCPU-1)")
    p.add_argument("--num_chunks",   type=int, default=None,
                   help="Total number of SLURM array chunks (omit for single job)")
    p.add_argument("--chunk_id",     type=int, default=None,
                   help="This task's chunk index (0-indexed, use with --num_chunks)")
    p.add_argument("--merge_only",   action="store_true",
                   help="Skip computation; just merge existing chunk files into graphs_all.pt")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    torch.multiprocessing.set_sharing_strategy("file_system")
    sys.stdout.reconfigure(line_buffering=True)  # flush on every newline
    print(f"Output dir : {args.output_dir}", flush=True)
    print(f"Workers    : {args.num_workers}", flush=True)
    print(f"MCS timeout: {args.mcs_timeout}s", flush=True)
    if args.num_chunks:
        print(f"Chunking   : chunk {args.chunk_id} of {args.num_chunks}", flush=True)
    print(f"ATOM_DIM={ATOM_DIM}  BOND_DIM={BOND_DIM}", flush=True)
    print(flush=True)

    if args.merge_only:
        if not args.num_chunks:
            raise ValueError("--merge_only requires --num_chunks")
        merge_chunks(args.output_dir, args.num_chunks)
        return

    df = load_and_clean(args.input, args.output_dir, args.sample_size, args.seed)

    precompute_all(
        df, args.output_dir,
        mcs_timeout=args.mcs_timeout,
        num_workers=args.num_workers,
        chunk_id=args.chunk_id,
        num_chunks=args.num_chunks,
    )

    print("\nAll done.", flush=True)
    print(f"  cleaned.pkl   → {os.path.join(args.output_dir, 'cleaned.pkl')}", flush=True)
    if not (args.num_chunks and args.num_chunks > 1):
        print(f"  graphs_all.pt → {os.path.join(args.output_dir, 'graphs_all.pt')}", flush=True)
    print("Pass --output_dir to hybrid_topo_rbfe.py's --graph_cache_dir to use these.", flush=True)


if __name__ == "__main__":
    main()
