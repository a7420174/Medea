#!/usr/bin/env python3
"""
Preprocess h5ad files for TranscriptFormer compatibility.

Usage:
    python -m medea.tool_space.tf_preprocess <input.h5ad> <output.h5ad>
"""

import os
import sys

import anndata as ad
import numpy as np
import scanpy as sc


def preprocess_adata(input_path: str, output_path: str) -> bool:
    """
    Validate and fix an h5ad file for TranscriptFormer inference.

    Fixes applied:
    - Duplicate var_names made unique
    - Missing 'ensembl_id' column added from var.index
    - Genes with zero expression removed

    Returns True on success, False on failure.
    """
    print(f"Starting preprocessing: {input_path}")
    print("=" * 70)

    if not os.path.exists(input_path):
        print(f"ERROR: Input file not found: {input_path}")
        return False

    try:
        adata = ad.read_h5ad(input_path)
        print(f"Loaded: {adata.shape[0]} cells x {adata.shape[1]} genes")
    except Exception as e:
        print(f"ERROR: Could not load AnnData file: {e}")
        return False

    original_shape = adata.shape

    # --- Diagnostics ---
    print("\nRunning diagnostics...")
    has_nan = np.isnan(adata.X.data).any() if hasattr(adata.X, "data") else np.isnan(adata.X).any()
    has_inf = np.isinf(adata.X.data).any() if hasattr(adata.X, "data") else np.isinf(adata.X).any()
    if has_nan or has_inf:
        print(f"  WARNING: NaN/Inf values found in expression matrix")

    has_dups = adata.var.index.nunique() < len(adata.var.index)
    missing_ensembl = "ensembl_id" not in adata.var.columns

    genes_before = adata.n_vars
    sc.pp.filter_genes(adata, min_cells=1)
    zero_genes = genes_before - adata.n_vars

    # Reload for clean fix pass
    adata = ad.read_h5ad(input_path)

    # --- Fixes ---
    print("\nApplying fixes...")
    if has_dups:
        adata.var_names_make_unique()
        print("  Made var_names unique")

    if missing_ensembl:
        adata.var["ensembl_id"] = adata.var.index
        print("  Added 'ensembl_id' column from var.index")

    if zero_genes > 0:
        sc.pp.filter_genes(adata, min_cells=1)
        print(f"  Removed {zero_genes} zero-expression genes")

    # --- Save ---
    try:
        adata.write(output_path)
    except Exception as e:
        print(f"ERROR: Could not save file: {e}")
        return False

    print(f"\nDone: {original_shape[0]} x {original_shape[1]} -> {adata.shape[0]} x {adata.shape[1]}")
    print(f"Saved: {output_path}")
    return True


def main():
    if len(sys.argv) != 3:
        print("Usage: python -m medea.tool_space.tf_preprocess <input.h5ad> <output.h5ad>")
        sys.exit(1)

    input_path, output_path = sys.argv[1], sys.argv[2]

    if os.path.abspath(input_path) == os.path.abspath(output_path):
        print("ERROR: Input and output paths cannot be the same.")
        sys.exit(1)

    if os.path.exists(output_path):
        ans = input(f"Output file already exists: {output_path}\nOverwrite? (y/N): ")
        if ans.lower() != "y":
            sys.exit(0)

    if not preprocess_adata(input_path, output_path):
        sys.exit(1)


if __name__ == "__main__":
    main()
