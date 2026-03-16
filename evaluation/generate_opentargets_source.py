#!/usr/bin/env python3
"""
Generate targetID source CSV files for cancer types using Open Targets API.

For each cancer type:
1. Query Open Targets GraphQL API for disease-associated targets with association scores
2. Sample 50 QA rows, each with 5 candidate genes (1 correct + 4 distractors)
3. Ground truth (y) = gene with highest overall association score among the 5 candidates

Usage:
    python evaluation/generate_opentargets_source.py --seed 42
    python evaluation/generate_opentargets_source.py --seed 42 --cancers "lung cancer,breast cancer"
    python evaluation/generate_opentargets_source.py --seed 42 --samples-per-disease 100
"""

import os
import sys
import json
import random
import argparse
import time
import requests
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Set

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Open Targets GraphQL API
OT_API_URL = "https://api.platform.opentargets.org/api/v4/graphql"

# Major cancer types with their common names for Open Targets search
DEFAULT_CANCER_TYPES = [
    "lung carcinoma",
    "breast carcinoma",
    "colorectal carcinoma",
    "hepatocellular carcinoma",
    "gastric carcinoma",
    "pancreatic carcinoma",
    "ovarian carcinoma",
    "prostate carcinoma",
    "melanoma",
    "glioblastoma",
    "renal cell carcinoma",
    "bladder carcinoma",
    "head and neck squamous cell carcinoma",
    "endometrial carcinoma",
    "acute myeloid leukemia",
    "chronic lymphocytic leukemia",
    "multiple myeloma",
    "non-Hodgkin lymphoma",
    "neuroblastoma",
    "osteosarcoma",
]


def get_efo_id(disease_name: str, max_retries: int = 3) -> Optional[str]:
    """Retrieve EFO/MONDO ID for a disease via EMBL-EBI OLS API."""
    api_url = "https://www.ebi.ac.uk/ols/api/search"

    for attempt in range(max_retries):
        try:
            # Try exact match first
            params = {'q': disease_name, 'ontology': 'efo', 'exact': 'true'}
            response = requests.get(api_url, params=params, timeout=15)
            if response.status_code == 200:
                results = response.json()
                if results['response']['numFound'] > 0:
                    obo_id = results['response']['docs'][0]['obo_id']
                    disease_id = obo_id.replace(":", "_")
                    if 'EFO' in disease_id or 'MONDO' in disease_id:
                        return disease_id

            # Fuzzy search
            params['exact'] = 'false'
            response = requests.get(api_url, params=params, timeout=15)
            if response.status_code == 200:
                results = response.json()
                for doc in results['response']['docs']:
                    obo_id = doc.get('obo_id', '')
                    disease_id = obo_id.replace(":", "_")
                    if 'EFO' in disease_id or 'MONDO' in disease_id:
                        return disease_id
        except Exception as e:
            print(f"  [Attempt {attempt+1}/{max_retries}] EFO lookup error: {e}")
            time.sleep(2)

    return None


def query_disease_targets_with_scores(
    efo_id: str,
    max_retries: int = 5
) -> List[Dict]:
    """
    Query Open Targets API for disease-associated targets WITH scores.

    Returns list of dicts: [{"symbol": "GENE", "score": 0.85, "datatype_scores": {...}}, ...]
    """
    query = """
    query diseaseTargets($efoId: String!, $size: Int!, $index: Int!) {
        disease(efoId: $efoId) {
            id
            name
            associatedTargets(page: {size: $size, index: $index}) {
                count
                rows {
                    target {
                        id
                        approvedSymbol
                    }
                    score
                    datatypeScores {
                        id
                        score
                    }
                }
            }
        }
    }
    """

    page_size = 3000
    page_index = 0
    all_targets = []

    while True:
        variables = {"efoId": efo_id, "size": page_size, "index": page_index}

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    OT_API_URL,
                    json={"query": query, "variables": variables},
                    timeout=30
                )
                response.raise_for_status()
                api_response = response.json()

                if "errors" in api_response:
                    raise ValueError(f"GraphQL errors: {api_response['errors']}")

                disease_data = api_response.get("data", {}).get("disease")
                if not disease_data:
                    return all_targets

                associated = disease_data.get("associatedTargets", {})
                rows = associated.get("rows", [])
                total_count = associated.get("count", 0)

                if page_index == 0:
                    print(f"  Total associations: {total_count}")

                for row in rows:
                    symbol = row.get("target", {}).get("approvedSymbol")
                    score = row.get("score", 0)
                    datatype_scores = {
                        ds.get("id", ""): ds.get("score", 0)
                        for ds in row.get("datatypeScores", [])
                    }

                    if symbol and score > 0:
                        all_targets.append({
                            "symbol": symbol,
                            "score": score,
                            "datatype_scores": datatype_scores
                        })

                # Check pagination
                if len(rows) < page_size or (page_index + 1) * page_size >= total_count:
                    return all_targets

                page_index += 1
                break

            except Exception as e:
                print(f"  [Attempt {attempt+1}/{max_retries}] API error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))
                else:
                    print(f"  Failed after {max_retries} attempts")
                    return all_targets

    return all_targets


def get_cancer_tissue_celltype(cancer_name: str) -> str:
    """Map cancer type to a representative tissue/cell type for the CSV."""
    mapping = {
        "lung": "lung epithelial cell",
        "breast": "breast epithelial cell",
        "colorectal": "intestinal epithelial cell",
        "hepatocellular": "hepatocyte",
        "gastric": "gastric epithelial cell",
        "pancreatic": "pancreatic ductal cell",
        "ovarian": "ovarian surface epithelial cell",
        "prostate": "prostate epithelial cell",
        "melanoma": "melanocyte",
        "glioblastoma": "astrocyte",
        "renal": "renal tubular cell",
        "bladder": "bladder urothelial cell",
        "head and neck": "squamous epithelial cell",
        "endometrial": "endometrial epithelial cell",
        "acute myeloid leukemia": "myeloid progenitor cell",
        "chronic lymphocytic leukemia": "b lymphocyte",
        "multiple myeloma": "plasma cell",
        "non-hodgkin": "b lymphocyte",
        "neuroblastoma": "neural crest cell",
        "osteosarcoma": "osteoblast",
    }

    cancer_lower = cancer_name.lower()
    for key, celltype in mapping.items():
        if key in cancer_lower:
            return celltype

    return "epithelial cell"


def generate_qa_samples(
    targets: List[Dict],
    disease_name: str,
    n_samples: int = 50,
    n_candidates: int = 5,
    seed: int = 42,
    score_type: str = "overall"
) -> List[Dict]:
    """
    Generate QA samples from target list.

    Strategy:
    - Sort targets by score
    - For each sample: pick 1 high-score gene (correct) + 4 lower-score genes (distractors)
    - Ensure the correct gene has the highest score among the 5 candidates
    """
    rng = random.Random(seed)

    if len(targets) < n_candidates:
        print(f"  WARNING: Only {len(targets)} targets available, need at least {n_candidates}")
        return []

    # Sort by score descending
    sorted_targets = sorted(targets, key=lambda x: x["score"], reverse=True)

    # Split into high-score pool (top 10%) and low-score pool (bottom 90%)
    split_idx = max(n_candidates, int(len(sorted_targets) * 0.1))
    high_pool = sorted_targets[:split_idx]
    low_pool = sorted_targets[split_idx:]

    # If low_pool is too small, adjust
    if len(low_pool) < n_candidates - 1:
        split_idx = max(1, len(sorted_targets) - (n_candidates - 1))
        high_pool = sorted_targets[:split_idx]
        low_pool = sorted_targets[split_idx:]

    celltype = get_cancer_tissue_celltype(disease_name)
    samples = []
    used_combinations = set()
    max_attempts = n_samples * 10
    attempts = 0

    while len(samples) < n_samples and attempts < max_attempts:
        attempts += 1

        # Pick 1 correct gene from high-score pool
        correct = rng.choice(high_pool)

        # Pick 4 distractors from low-score pool (must have lower score)
        available_distractors = [t for t in low_pool if t["symbol"] != correct["symbol"] and t["score"] < correct["score"]]

        if len(available_distractors) < n_candidates - 1:
            # Fallback: use all targets except correct
            available_distractors = [t for t in sorted_targets if t["symbol"] != correct["symbol"] and t["score"] < correct["score"]]

        if len(available_distractors) < n_candidates - 1:
            continue

        distractors = rng.sample(available_distractors, n_candidates - 1)

        # Create candidate list and shuffle
        candidates = [correct] + distractors
        rng.shuffle(candidates)

        candidate_symbols = [c["symbol"] for c in candidates]
        combo_key = tuple(sorted(candidate_symbols))

        if combo_key in used_combinations:
            continue
        used_combinations.add(combo_key)

        # Verify correct gene has highest score among candidates
        max_score_gene = max(candidates, key=lambda x: x["score"])
        if max_score_gene["symbol"] != correct["symbol"]:
            continue

        samples.append({
            "disease": disease_name,
            "celltype": celltype,
            "candidate_genes": str(candidate_symbols),
            "y": correct["symbol"]
        })

    if len(samples) < n_samples:
        print(f"  WARNING: Only generated {len(samples)}/{n_samples} samples (insufficient unique combinations)")

    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Generate targetID source CSVs for cancer types from Open Targets API"
    )
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument(
        '--cancers', type=str, default=None,
        help='Comma-separated list of cancer names (default: all major cancer types)'
    )
    parser.add_argument(
        '--samples-per-disease', type=int, default=50,
        help='Number of QA samples per disease (default: 50)'
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Output directory (default: evaluation/targetID/source/)'
    )
    args = parser.parse_args()

    # Determine cancer list
    if args.cancers:
        cancer_list = [c.strip() for c in args.cancers.split(",")]
    else:
        cancer_list = DEFAULT_CANCER_TYPES

    # Output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = args.output_dir or os.path.join(script_dir, "targetID", "source")
    os.makedirs(output_dir, exist_ok=True)

    print(f"{'='*60}")
    print(f"Open Targets Source CSV Generator")
    print(f"{'='*60}")
    print(f"Cancer types: {len(cancer_list)}")
    print(f"Samples per disease: {args.samples_per_disease}")
    print(f"Seed: {args.seed}")
    print(f"Output dir: {output_dir}")
    print(f"{'='*60}\n")

    summary = []

    for cancer_name in cancer_list:
        print(f"\n[{cancer_name}] Looking up EFO ID...")
        efo_id = get_efo_id(cancer_name)

        if not efo_id:
            print(f"  SKIPPED: Could not find EFO/MONDO ID for '{cancer_name}'")
            summary.append({"disease": cancer_name, "status": "SKIPPED (no EFO ID)", "samples": 0})
            continue

        print(f"  EFO ID: {efo_id}")
        print(f"  Querying targets...")

        targets = query_disease_targets_with_scores(efo_id)

        if len(targets) < 10:
            print(f"  SKIPPED: Only {len(targets)} targets found (need >= 10)")
            summary.append({"disease": cancer_name, "status": f"SKIPPED ({len(targets)} targets)", "samples": 0})
            continue

        print(f"  Retrieved {len(targets)} targets with scores")

        # Generate QA samples
        samples = generate_qa_samples(
            targets=targets,
            disease_name=cancer_name,
            n_samples=args.samples_per_disease,
            seed=args.seed
        )

        if not samples:
            print(f"  SKIPPED: Could not generate any samples")
            summary.append({"disease": cancer_name, "status": "SKIPPED (no samples)", "samples": 0})
            continue

        # Save to CSV
        # Sanitize filename
        safe_name = cancer_name.lower().replace(" ", "_").replace("'", "").replace("-", "_")
        output_file = os.path.join(output_dir, f"targetid-{safe_name}-{args.seed}.csv")

        df = pd.DataFrame(samples)
        df.to_csv(output_file, index=False)

        print(f"  Saved {len(samples)} samples to {output_file}")
        summary.append({"disease": cancer_name, "status": "OK", "samples": len(samples)})

        # Small delay between API calls
        time.sleep(1)

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for s in summary:
        print(f"  {s['disease']:<45} {s['status']:<30} {s['samples']} samples")

    total_samples = sum(s['samples'] for s in summary)
    successful = sum(1 for s in summary if s['status'] == 'OK')
    print(f"\nTotal: {successful}/{len(cancer_list)} diseases, {total_samples} samples")


if __name__ == "__main__":
    main()
