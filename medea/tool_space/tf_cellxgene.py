#!/usr/bin/env python3
"""
Download disease scRNA-seq data from CellxGene Census.

Requires:
    pip install cellxgene-census

Usage (Python):
    from medea.tool_space.tf_cellxgene import download_disease_h5ad
    path = download_disease_h5ad("rheumatoid arthritis", "/path/to/Disease-atlas")

Usage (CLI):
    python -m medea.tool_space.tf_cellxgene "rheumatoid arthritis" /path/to/Disease-atlas
"""

import os
import sys

from thefuzz import process as fuzz_process


def _get_census():
    try:
        import cellxgene_census
    except ImportError:
        raise ImportError(
            "cellxgene-census is not installed.\n"
            "Install it with: pip install cellxgene-census"
        )
    return cellxgene_census


def list_available_diseases() -> list[str]:
    """Return all unique disease values available in CellxGene Census (human)."""
    cx = _get_census()
    print("Querying CellxGene Census for available diseases...")
    with cx.open_soma() as census:
        obs = census["census_data"]["homo_sapiens"].obs.read(
            column_names=["disease"]
        ).concat().to_pandas()
    diseases = sorted(obs["disease"].dropna().unique().tolist())
    print(f"Found {len(diseases)} unique disease values")
    return diseases


def resolve_disease_name(disease_name: str, threshold: int = 80) -> str:
    """
    Resolve a user-provided disease name to the exact CellxGene Census value.

    Uses fuzzy matching if the exact name is not found.

    Args:
        disease_name: Disease name to resolve (e.g. "rheumatoid arthritis").
        threshold: Minimum fuzzy match score (0-100) to auto-accept.

    Returns:
        Exact disease name as used in CellxGene Census.

    Raises:
        ValueError: If no match above threshold is found.
    """
    available = list_available_diseases()

    # Exact match (case-insensitive)
    lower = disease_name.lower()
    for d in available:
        if d.lower() == lower:
            return d

    # Fuzzy match
    matches = fuzz_process.extract(disease_name, available, limit=5)
    best_name, best_score = matches[0]

    if best_score >= threshold:
        if best_score < 95:
            print(f"  '{disease_name}' matched to '{best_name}' (score: {best_score})")
        return best_name

    print(f"  No confident match for '{disease_name}'. Top candidates:")
    for name, score in matches:
        print(f"    {score:3d}  {name}")
    raise ValueError(
        f"Could not confidently match '{disease_name}' to a CellxGene disease name "
        f"(best: '{best_name}', score: {best_score} < {threshold}).\n"
        f"Call list_available_diseases() to browse options."
    )


def download_disease_h5ad(
    disease_name: str,
    atlas_root: str,
    max_cells: int | None = None,
) -> str:
    """
    Download scRNA-seq data for a disease from CellxGene Census and save as h5ad.

    The file is saved to:
        {atlas_root}/{disease_key}/{disease_key}.h5ad

    Args:
        disease_name: Human-readable disease name (fuzzy-matched to Census values).
        atlas_root: Root directory for the Disease-atlas (DISEASE_ATLAS_PATH).
        max_cells: If set, randomly subsample to this many cells (useful for testing).

    Returns:
        Path to the saved h5ad file.
    """
    cx = _get_census()

    # Resolve to exact Census disease name
    census_disease = resolve_disease_name(disease_name)
    print(f"Downloading '{census_disease}' from CellxGene Census...")

    disease_key = disease_name.lower().replace(" ", "_")
    out_dir = os.path.join(atlas_root, disease_key)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{disease_key}.h5ad")

    if os.path.exists(out_path):
        print(f"  Already exists: {out_path}")
        return out_path

    obs_filter = f"disease == '{census_disease}' and is_primary_data == True"
    print(f"  Filter: {obs_filter}")

    with cx.open_soma() as census:
        adata = cx.get_anndata(
            census,
            organism="Homo sapiens",
            obs_value_filter=obs_filter,
        )

    print(f"  Downloaded: {adata.n_obs:,} cells x {adata.n_vars:,} genes")

    if max_cells is not None and adata.n_obs > max_cells:
        import numpy as np
        rng = np.random.default_rng(42)
        idx = rng.choice(adata.n_obs, size=max_cells, replace=False)
        adata = adata[idx].copy()
        print(f"  Subsampled to {adata.n_obs:,} cells")

    adata.write_h5ad(out_path)
    size_mb = os.path.getsize(out_path) / (1024 ** 2)
    print(f"  Saved: {out_path} ({size_mb:.1f} MB)")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Download disease scRNA-seq data from CellxGene Census."
    )
    parser.add_argument("disease_name", help="Disease name (e.g. 'rheumatoid arthritis')")
    parser.add_argument(
        "atlas_root",
        nargs="?",
        default=os.environ.get("DISEASE_ATLAS_PATH", "Disease-atlas"),
        help="Root directory for Disease-atlas (default: $DISEASE_ATLAS_PATH or ./Disease-atlas)",
    )
    parser.add_argument("--max-cells", type=int, default=None,
                        help="Subsample to N cells (for testing)")
    parser.add_argument("--list-diseases", action="store_true",
                        help="List all available disease names and exit")
    args = parser.parse_args()

    if args.list_diseases:
        diseases = list_available_diseases()
        for d in diseases:
            print(d)
        return

    try:
        path = download_disease_h5ad(args.disease_name, args.atlas_root, args.max_cells)
        print(f"\nSUCCESS: {path}")
    except Exception as e:
        import traceback
        print(f"\nERROR: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
