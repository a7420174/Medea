#!/usr/bin/env python3
"""
Convert celltype_disease_cge_inference output (.h5ad) into a fast embedding store.

Output structure:
    embedding_store/{disease_name}/
    ├── metadata.json.gz
    └── {cell_type}_{state}.npy   (float32, shape: [n_genes, embedding_dim])

Usage:
    python -m medea.tool_space.tf_embedding_store <cge_output.h5ad> <disease_name> [--store-dir <path>]
"""

import argparse
import gzip
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import anndata as ad
import mygene
import numpy as np


def build_gene_maps(ensembl_ids: list[str]) -> tuple[dict, dict]:
    """Query MyGene.info to build symbol <-> Ensembl ID maps."""
    print(f"Querying MyGene.info for {len(ensembl_ids):,} Ensembl IDs...")
    mg = mygene.MyGeneInfo()
    results = mg.querymany(
        ensembl_ids,
        scopes="ensembl.gene",
        fields="symbol,ensembl.gene",
        species="human",
        returnall=False,
        verbose=False,
    )

    symbol_to_ensembl: dict[str, str] = {}
    ensembl_to_symbol: dict[str, str] = {}
    for hit in results:
        if "notfound" in hit or not hit.get("symbol"):
            continue
        ens = hit["query"].upper()
        sym = hit["symbol"].upper()
        symbol_to_ensembl[sym] = ens
        ensembl_to_symbol[ens] = sym

    print(f"  Mapped {len(symbol_to_ensembl):,} / {len(ensembl_ids):,} genes")
    unmapped = len(ensembl_ids) - len(symbol_to_ensembl)
    if unmapped:
        print(f"  WARNING: {unmapped} genes could not be mapped to a symbol")
    return symbol_to_ensembl, ensembl_to_symbol


def prepare_embedding_store(cge_h5ad_path: str, disease_name: str, store_dir: str) -> None:
    """
    Convert a CGE inference h5ad into a per-group .npy embedding store.

    Args:
        cge_h5ad_path: Output of celltype_disease_cge_inference().
        disease_name: Human-readable disease name (e.g. "systemic lupus erythematosus").
        store_dir: Root embedding store directory.
    """
    print(f"\nPreparing embedding store")
    print(f"  Source  : {cge_h5ad_path}")
    print(f"  Disease : {disease_name}")
    print(f"  Store   : {store_dir}")
    print("=" * 70)

    adata = ad.read_h5ad(cge_h5ad_path)

    if "celltype_disease_embeddings" not in adata.uns:
        raise KeyError(
            "'celltype_disease_embeddings' not found in adata.uns. "
            "Pass the output of celltype_disease_cge_inference()."
        )

    group_embeddings: dict[str, np.ndarray] = adata.uns["celltype_disease_embeddings"]
    group_counts: dict[str, np.ndarray] = adata.uns.get("celltype_disease_counts", {})
    group_info: dict = adata.uns.get("group_info", {})
    gene_names: list[str] = list(adata.uns["gene_names"])
    embedding_dim: int = int(
        adata.uns.get("embedding_dimensions", next(iter(group_embeddings.values())).shape[1])
    )
    print(f"  {len(gene_names):,} genes | {len(group_embeddings)} groups | dim={embedding_dim}")

    symbol_to_ensembl, ensembl_to_symbol = build_gene_maps(gene_names)

    disease_key = disease_name.lower().replace(" ", "_")
    out_dir = Path(store_dir) / disease_key
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save per-group .npy files
    groups_meta: dict[str, dict] = {}
    print("\nSaving .npy files...")
    for group_name, emb_matrix in group_embeddings.items():
        assert emb_matrix.shape == (len(gene_names), embedding_dim), (
            f"Unexpected shape {emb_matrix.shape} for group '{group_name}'"
        )
        np.save(out_dir / f"{group_name}.npy", emb_matrix.astype(np.float32))

        if group_name in group_info:
            cell_type = group_info[group_name]["cell_type"].lower().replace(" ", "_")
            disease_state = group_info[group_name]["disease_state"]
            total_cells = group_info[group_name].get("total_cells", 0)
        else:
            # Fallback: parse group_name as {cell_type}_{disease_state}
            parts = group_name.rsplit("_", 1)
            cell_type = parts[0] if len(parts) == 2 else group_name
            disease_state = parts[1] if len(parts) == 2 else "unknown"
            total_cells = int(group_counts[group_name].sum()) if group_name in group_counts else 0

        groups_meta[group_name] = {
            "cell_type": cell_type,
            "disease_state": disease_state,
            "group_name": group_name,
            "total_cells": total_cells,
        }
        print(f"  {group_name}.npy  ({total_cells:,} cells)")

    # Save metadata
    metadata = {
        "description": f"TranscriptFormer CGE embeddings for {disease_name}",
        "source_file": os.path.basename(cge_h5ad_path),
        "creation_date": datetime.now().isoformat(),
        "embedding_dimensions": embedding_dim,
        "ensembl_ids_ordered": gene_names,
        "groups": groups_meta,
        "gene_map_symbol_to_ensembl": symbol_to_ensembl,
        "gene_map_ensembl_to_symbol": ensembl_to_symbol,
    }
    metadata_path = out_dir / "metadata.json.gz"
    with gzip.open(metadata_path, "wt", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False)

    size_kb = metadata_path.stat().st_size / 1024
    print(f"\nmetadata.json.gz saved ({size_kb:.1f} KB)")
    print(f"Embedding store ready: {out_dir}")
    print(f"\nUsage:")
    print(f"  tool.get_embedding_for_context(state, cell_type, genes, disease='{disease_name}')")


def main():
    parser = argparse.ArgumentParser(
        description="Convert CGE inference output to TranscriptFormer embedding store."
    )
    parser.add_argument("cge_h5ad", help="Path to cge_output.h5ad")
    parser.add_argument("disease_name", help="Disease name (e.g. 'systemic lupus erythematosus')")
    parser.add_argument(
        "--store-dir",
        default=os.path.join(
            os.environ.get("MEDEADB_PATH", "."),
            "transcriptformer_embedding",
        ),
        help="Root embedding store directory (default: $MEDEADB_PATH/transcriptformer_embedding)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.cge_h5ad):
        print(f"ERROR: File not found: {args.cge_h5ad}")
        sys.exit(1)

    try:
        prepare_embedding_store(
            cge_h5ad_path=os.path.abspath(args.cge_h5ad),
            disease_name=args.disease_name,
            store_dir=args.store_dir,
        )
    except Exception as e:
        import traceback
        print(f"\nERROR: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
