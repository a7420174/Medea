#!/usr/bin/env python3
"""
Cell-type and disease-state specific CGE inference using TranscriptFormer.

Generates per-(cell_type, disease_state) averaged gene embeddings from an h5ad file.

Usage:
    python -m medea.tool_space.tf_inference <input.h5ad> <output.h5ad> <checkpoint_dir> [chunk_size] [batch_size] [n_jobs]
"""

import gc
import os
import re
import shutil
import subprocess
import sys
import tempfile

import anndata as ad
import numpy as np
import pandas as pd
from joblib import Parallel, delayed


# ---------------------------------------------------------------------------
# Disk / chunk helpers
# ---------------------------------------------------------------------------

def _check_disk_space_gb() -> float:
    try:
        return shutil.disk_usage("/").free / (1024 ** 3)
    except Exception:
        return 0.0


def _safe_group_name(cell_type: str, disease: str) -> str:
    ct = cell_type.replace(" ", "_").replace("-", "_").replace(".", "_").replace(",", "_").lower()
    ds = disease.replace(" ", "_").replace("-", "_").replace(".", "_").replace(",", "_").lower()
    return re.sub(r"[^\w.-]+", "_", f"{ct}_{ds}")


# ---------------------------------------------------------------------------
# Cell group analysis
# ---------------------------------------------------------------------------

def _analyze_cell_groups(adata: ad.AnnData) -> pd.DataFrame:
    groups = (
        adata.obs.groupby(["cell_type", "disease"])
        .size()
        .reset_index(name="cell_count")
        .sort_values("cell_count", ascending=False)
    )
    print(f"Found {len(groups)} cell-type x disease combinations:")
    for _, row in groups.iterrows():
        pct = row["cell_count"] / adata.n_obs * 100
        print(f"  {row['cell_type']} ({row['disease']}): {row['cell_count']:,} cells ({pct:.1f}%)")
    return groups


def _get_cell_indices(adata: ad.AnnData, cell_type: str, disease: str) -> np.ndarray:
    mask = (adata.obs["cell_type"] == cell_type) & (adata.obs["disease"] == disease)
    return np.where(mask)[0]


# ---------------------------------------------------------------------------
# Chunk creation and inference
# ---------------------------------------------------------------------------

def _create_chunk(adata: ad.AnnData, cell_indices: np.ndarray,
                  start: int, end: int, chunk_path: str) -> int:
    chunk = adata[cell_indices[start:end], :].copy()
    chunk.write(chunk_path)
    size_mb = os.path.getsize(chunk_path) / (1024 ** 2)
    print(f"    Chunk: {size_mb:.1f} MB ({len(chunk)} cells)")
    return len(chunk)


def _run_tf_inference(chunk_path: str, output_path: str,
                      checkpoint_path: str, batch_size: int = 1,
                      max_retries: int = 3) -> bool:
    """Run `transcriptformer inference` CLI on a single chunk."""
    output_dir = os.path.dirname(output_path)
    current_batch_size = batch_size

    for attempt in range(max_retries):
        cmd = [
            "transcriptformer", "inference",
            "--checkpoint-path", os.path.abspath(checkpoint_path),
            "--data-file", os.path.abspath(chunk_path),
            "--emb-type", "cge",
            "--output-filename", os.path.basename(output_path),
            "--batch-size", str(current_batch_size),
            "--precision", "16-mixed",
        ]
        print(f"    Attempt {attempt + 1}/{max_retries} (batch_size={current_batch_size})")

        process = subprocess.Popen(
            cmd, cwd=output_dir,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        lines = []
        try:
            for line in process.stdout:
                print(f"    | {line.rstrip()}")
                lines.append(line)
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            return False

        rc = process.wait()
        if rc != 0:
            if rc == -9:  # OOM
                current_batch_size = max(1, current_batch_size // 2)
                print(f"    OOM — retrying with batch_size={current_batch_size}")
                continue
            print(f"    Inference failed (exit code {rc})")
            return False

        # Locate output file (transcriptformer may place it in inference_results/)
        candidates = [
            output_path,
            os.path.join(output_dir, os.path.basename(output_path)),
            os.path.join(output_dir, "inference_results", os.path.basename(output_path)),
        ]
        for cand in candidates:
            if os.path.exists(cand):
                if cand != output_path:
                    shutil.move(cand, output_path)
                # Clean up inference_results dir if left behind
                inf_dir = os.path.join(output_dir, "inference_results")
                if os.path.exists(inf_dir):
                    shutil.rmtree(inf_dir)
                return True

        print(f"    Output not found after inference")
        return False

    print(f"    All {max_retries} attempts failed")
    return False


# ---------------------------------------------------------------------------
# Running average merge
# ---------------------------------------------------------------------------

def _load_chunk_embeddings(chunk_path: str):
    """Return (embeddings, gene_names) from a chunk h5ad, or (None, None)."""
    try:
        chunk = ad.read_h5ad(chunk_path)
        if "cge_embeddings" not in chunk.uns:
            return None, None
        emb = chunk.uns["cge_embeddings"]
        genes = chunk.uns.get("cge_gene_names")
        print(f"    Loaded: {emb.shape[0]} embeddings x {emb.shape[1]} dims")
        return emb, genes
    except Exception as e:
        print(f"    Error loading embeddings: {e}")
        return None, None


def _merge_embeddings(running_avg: dict | None, running_counts: dict | None,
                      new_embeddings: np.ndarray, new_gene_names: list):
    """Incrementally update running per-gene average."""
    if running_avg is None:
        running_avg, running_counts = {}, {}
    if not len(new_gene_names):
        return running_avg, running_counts

    df = pd.DataFrame({"gene": new_gene_names, "embedding": list(new_embeddings)})
    agg = df.groupby("gene")["embedding"].agg(["sum", "count"])

    for gene, row in agg.iterrows():
        gene = str(gene)
        if gene in running_avg:
            old_n = running_counts[gene]
            new_n = old_n + row["count"]
            running_avg[gene] = running_avg[gene] * (old_n / new_n) + row["sum"] / new_n
            running_counts[gene] = new_n
        else:
            running_avg[gene] = row["sum"] / row["count"]
            running_counts[gene] = row["count"]

    return running_avg, running_counts


# ---------------------------------------------------------------------------
# Per-group processing
# ---------------------------------------------------------------------------

CHUNK_SIZE = 1000


def _process_group(adata: ad.AnnData, cell_type: str, disease: str,
                   cell_indices: np.ndarray, checkpoint_path: str,
                   temp_dir: str, batch_size: int = 1):
    """Process one (cell_type, disease) group and return running average embeddings."""
    print(f"\nProcessing: {cell_type} / {disease} ({len(cell_indices)} cells)")

    if len(cell_indices) == 0:
        return None, None, 0

    n_chunks = int(np.ceil(len(cell_indices) / CHUNK_SIZE))
    running_avg, running_counts = None, None
    total_cells, successful = 0, 0
    group_name = _safe_group_name(cell_type, disease)

    for i in range(n_chunks):
        if _check_disk_space_gb() < 3:
            print("  STOPPING: Low disk space")
            break

        start, end = i * CHUNK_SIZE, min((i + 1) * CHUNK_SIZE, len(cell_indices))
        if n_chunks > 1:
            print(f"  Chunk {i + 1}/{n_chunks} ({end - start} cells)")

        chunk_in = os.path.join(temp_dir, f"{group_name}_chunk{i}_input.h5ad")
        chunk_out_dir = os.path.join(temp_dir, f"{group_name}_chunk{i}_output")
        chunk_out = os.path.join(chunk_out_dir, f"{group_name}_chunk{i}_cge.h5ad")
        os.makedirs(chunk_out_dir, exist_ok=True)

        n_cells = _create_chunk(adata, cell_indices, start, end, chunk_in)
        ok = _run_tf_inference(chunk_in, chunk_out, checkpoint_path, batch_size)

        if ok and os.path.exists(chunk_out):
            emb, genes = _load_chunk_embeddings(chunk_out)
            if emb is not None and genes is not None:
                running_avg, running_counts = _merge_embeddings(running_avg, running_counts, emb, genes)
                total_cells += n_cells
                successful += 1
                print(f"  Chunk {i + 1} merged ({len(running_avg)} genes running)")
            else:
                print(f"  Chunk {i + 1}: no valid embeddings")
        else:
            print(f"  Chunk {i + 1} failed")

        if os.path.exists(chunk_in):
            os.remove(chunk_in)
        if os.path.exists(chunk_out_dir):
            shutil.rmtree(chunk_out_dir)
        gc.collect()

    print(f"  Done: {successful}/{n_chunks} chunks, {total_cells} cells")
    return (running_avg, running_counts, total_cells) if running_avg else (None, None, 0)


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def _save_embeddings(all_group_embeddings: dict, output_path: str,
                     original_adata: ad.AnnData) -> str:
    """Assemble per-group embeddings into a single h5ad."""
    print("\nSaving embeddings...")

    all_genes = sorted(set.union(*(set(d[0].keys()) for d in all_group_embeddings.values() if d[0])))
    gene_to_idx = {g: i for i, g in enumerate(all_genes)}

    group_keys = list(all_group_embeddings.keys())

    # Unique group names
    group_names, seen = [], set()
    for ct, ds in group_keys:
        base = _safe_group_name(ct, ds)
        name, n = base, 1
        while name in seen:
            name = f"{base}_{n}"; n += 1
        seen.add(name)
        group_names.append(name)

    first_valid = next((d for d in all_group_embeddings.values() if d[0]), None)
    if first_valid is None:
        raise RuntimeError("No valid embeddings to save")
    embedding_dim = len(next(iter(first_valid[0].values())))

    n_genes, n_groups = len(all_genes), len(group_keys)
    all_embeddings = np.zeros((n_groups, n_genes, embedding_dim), dtype=np.float32)
    all_counts = np.zeros((n_groups, n_genes), dtype=np.int32)

    var_data = pd.DataFrame({
        "cell_type": [k[0] for k in group_keys],
        "disease_state": [k[1] for k in group_keys],
        "group_name": group_names,
        "total_cells": [all_group_embeddings[k][2] for k in group_keys],
    }, index=group_names)

    for i, key in enumerate(group_keys):
        avg, counts, _ = all_group_embeddings[key]
        if not avg:
            continue
        genes = list(avg.keys())
        idxs = [gene_to_idx[g] for g in genes]
        all_embeddings[i, idxs] = np.array(list(avg.values()))
        all_counts[i, idxs] = np.array(list(counts.values()))

    final = ad.AnnData(
        X=np.zeros((n_genes, n_groups)),
        obs=pd.DataFrame(index=all_genes),
        var=var_data,
    )
    final.uns["celltype_disease_embeddings"] = {
        name: all_embeddings[i] for i, name in enumerate(group_names)
    }
    final.uns["celltype_disease_counts"] = {
        name: all_counts[i] for i, name in enumerate(group_names)
    }
    final.uns["gene_names"] = all_genes
    final.uns["group_info"] = var_data.to_dict("index")
    final.uns["original_data_shape"] = list(original_adata.shape)
    final.uns["averaging_method"] = "celltype_disease_specific"
    final.uns["embedding_dimensions"] = embedding_dim

    final.write(output_path)
    size_mb = os.path.getsize(output_path) / (1024 ** 2)
    print(f"Saved: {output_path} ({size_mb:.1f} MB)")
    print(f"  {n_genes:,} genes | {n_groups} groups | dim={embedding_dim}")
    return output_path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def celltype_disease_cge_inference(input_path: str, output_path: str,
                                    checkpoint_path: str,
                                    chunk_size: int = 1000,
                                    batch_size: int = 1,
                                    n_jobs: int = -1) -> str:
    """
    Run TranscriptFormer CGE inference for all cell-type x disease combinations.

    Args:
        input_path: Preprocessed h5ad file (output of tf_preprocess.py).
        output_path: Output h5ad path.
        checkpoint_path: TranscriptFormer checkpoint directory.
        chunk_size: Cells per inference chunk (default 1000).
        batch_size: Initial batch size for TranscriptFormer (auto-reduced on OOM).
        n_jobs: Parallel workers (-1 = all CPUs).

    Returns:
        Path to the saved output h5ad.
    """
    print(f"CGE Inference")
    print(f"  Input     : {input_path}")
    print(f"  Output    : {output_path}")
    print(f"  Checkpoint: {checkpoint_path}")
    print("=" * 70)

    if _check_disk_space_gb() < 10:
        raise RuntimeError("Insufficient disk space (need at least 10 GB free)")

    adata = ad.read_h5ad(input_path)
    print(f"Loaded: {adata.shape[0]} cells x {adata.shape[1]} genes")

    groups_df = _analyze_cell_groups(adata)
    output_dir = os.path.dirname(os.path.abspath(output_path))
    temp_dir = tempfile.mkdtemp(prefix="medea_tf_cge_", dir=output_dir)

    try:
        tasks = [
            delayed(_process_group)(
                adata,
                row["cell_type"], row["disease"],
                _get_cell_indices(adata, row["cell_type"], row["disease"]),
                checkpoint_path, temp_dir, batch_size,
            )
            for _, row in groups_df.iterrows()
        ]
        results = Parallel(n_jobs=n_jobs)(tasks)

        all_group_embeddings = {
            (row["cell_type"], row["disease"]): result
            for (_, row), result in zip(groups_df.iterrows(), results)
            if result[0] is not None
        }

        if not all_group_embeddings:
            raise RuntimeError("No valid group embeddings were produced")

        print(f"\nProcessed {len(all_group_embeddings)} groups successfully")
        return _save_embeddings(all_group_embeddings, output_path, adata)

    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 4:
        print("Usage: python -m medea.tool_space.tf_inference "
              "<input.h5ad> <output.h5ad> <checkpoint_dir> "
              "[chunk_size=1000] [batch_size=1] [n_jobs=-1]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    checkpoint_path = sys.argv[3]
    chunk_size = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
    batch_size = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    n_jobs = int(sys.argv[6]) if len(sys.argv) > 6 else -1

    try:
        result = celltype_disease_cge_inference(
            os.path.abspath(input_path),
            os.path.abspath(output_path),
            os.path.abspath(checkpoint_path),
            chunk_size, batch_size, n_jobs,
        )
        print(f"\nSUCCESS: {result}")
    except Exception as e:
        import traceback
        print(f"\nERROR: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
