#!/usr/bin/env python3
"""
This script distributes a sampled set of questions to N labellers.
Each question is labelled R times, and 'Foreign Words' questions
are only sent to labellers who speak the target language.
"""

import json  # for reading/writing JSON and JSONL files
import random  # for random sampling and shuffling
import argparse  # for parsing CLI arguments
import os  # for filesystem operations
from collections import defaultdict  # for grouping items in bins


def parse_args():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace with the following attributes:
        - input_files: list[str], paths to prediction JSONL files
        - labellers_file: str, path to labellers JSON metadata
        - output_dir: str, directory for output workload files
        - samples_per_bin: int, samples per bin per file (default=2)
        - labels_per_question: int, assignments per question (default=3)
        - seed: int, random seed for reproducibility
    """
    parser = argparse.ArgumentParser(
        description="Distribute questions to labellers."
    )
    parser.add_argument(
        "--input-files", nargs="+", required=True,
        help="Paths to the 8 prediction JSONL files."
    )
    parser.add_argument(
        "--labellers-file", required=True,
        help="JSON file mapping usernames to {'email':…, 'language':[…]}"
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory to write per-labeller workload files."
    )
    parser.add_argument(
        "--samples-per-bin", type=int, default=2,
        help="Number of samples per (category/depth) bin per file (default=2)."
    )
    parser.add_argument(
        "--labels-per-question", type=int, default=3,
        help="Number of distinct labellers per question (default=3)."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default=42)."
    )
    return parser.parse_args()


def load_labellers(path):
    """
    Load labeller metadata from a JSON file.

    Args:
        path: str, path to the JSON file containing a list of
              { 'username': str, 'email': str, 'language': [str, ...] }

    Returns:
        dict: mapping username -> {'email':..., 'language':[...]}.
    """
    with open(path, "r") as f:
        entries = json.load(f)
    # Convert list of dicts into username-keyed map
    return { e['username']: {'email': e['email'], 'language': e.get('language', [])}
             for e in entries }


def sample_questions(files, per_bin, labeller_langs):
    """
    Sample questions from each prediction file into bins:
    1. Non-foreign questions: grouped by (category, depth).
    2. Foreign-word questions: grouped by (language, depth)
       for each language spoken by any labeller.

    Args:
        files: list[str], input JSONL prediction files.
        per_bin: int, max number of samples per bin.
        labeller_langs: set[str], lowercased languages spoken by labellers.

    Returns:
        List[dict]: sampled question objects.
    """
    sampled = []
    for fn in files:
        # Bins for normal categories
        normal_bins = defaultdict(list)
        # Pre-build foreign bins for each (lang, depth)
        foreign_bins = { (lang, depth): []
                         for lang in labeller_langs
                         for depth in [0, 1, 2, 3] }
        
        # print("foreign_bins:", foreign_bins)

        # Read and assign each question to its bin
        with open(fn, 'r') as f:
            for line in f:
                q = json.loads(line)
                cat = q['category']
                depth = q['evolution_depth']
                if cat == 'Foreign Words':
                    lang = q['misc']['foreign_language']
                    # Only if language-depth bin exists
                    if (lang, depth) in foreign_bins:
                        foreign_bins[(lang, depth)].append(q)
                else:
                    normal_bins[(cat, depth)].append(q)

        # Sample up to per_bin from each normal bin
        cur_len = len(sampled)
        for group in normal_bins.values():
            sampled.extend(random.sample(group, per_bin))
        print(f"Sampled {len(sampled) - cur_len} questions normal category")

        # Sample up to per_bin from each foreign-language bin
        cur_len = len(sampled)
        for group in foreign_bins.values():
            count = min(len(group), per_bin)
            if count > 0:
                sampled.extend(random.sample(group, count))
        print(f"Sampled {len(sampled) - cur_len} questions foreign words category")

    return sampled


def distribute(qs, labellers, labels_per_q):
    """
    Assign each sampled question to labellers:
    - Use a greedy least-loaded approach.
    - Enforce language constraint for Foreign Words questions.

    Args:
        qs: list[dict], sampled questions.
        labellers: dict, username -> info.
        labels_per_q: int, desired labels per question.

    Returns:
        workloads: dict, username -> list of questions.
        counts: dict, username -> total assigned count.
    """
    users = list(labellers.keys())
    counts = {u: 0 for u in users}
    workloads = {u: [] for u in users}

    random.shuffle(qs)  # randomize processing order
    for q in qs:
        if q.get('category') == 'Foreign Words':
            lang = q['misc']['foreign_language']
            # Only labellers with matching specialty
            eligible = [u for u,info in labellers.items()
                        if lang in [l for l in info['language']]]
        else:
            # All labellers eligible for non-foreign categories
            eligible = users.copy()

        if not eligible:
            print(f"foreign language: {lang}")
            raise RuntimeError(f"No eligible labeller for question ID {q.get('unique_id_eval')}")

        # Sort by current load
        eligible.sort(key=lambda u: counts[u])
        # Pick up to labels_per_q least-loaded workers
        chosen = eligible[:min(labels_per_q, len(eligible))]

        # Record assignments
        for u in chosen:
            workloads[u].append(q)
            counts[u] += 1

        # Warn only for non-foreign if shortage
        if len(chosen) < labels_per_q and q.get('category') != 'Foreign Words':
            print(f"⚠️ Only {len(chosen)} labellers for question {q.get('unique_id_eval')} (needed {labels_per_q})")

    return workloads, counts


def write_workloads(workloads, outdir):
    """
    Write each labeller's assigned questions to a JSONL file.

    Args:
        workloads: dict, username -> list of questions.
        outdir: str, target directory.
    """
    os.makedirs(outdir, exist_ok=True)
    for user, qs in workloads.items():
        path = os.path.join(outdir, f"{user}_workload.jsonl")
        with open(path, 'w') as f:
            for q in qs:
                f.write(json.dumps(q) + "\n")


def write_manifest(workloads, outdir):
    """
    Generate a manifest.json mapping users -> their workload file.
    """
    manifest = { user: os.path.join(outdir, f"{user}_workload.jsonl")
                 for user in workloads }
    with open(os.path.join(outdir, 'manifest.json'), 'w') as mf:
        json.dump(manifest, mf, indent=2)


def main():
    # Parse CLI args and seed RNG
    args = parse_args()
    random.seed(args.seed)

    # Load labeller info and collect all spoken languages
    labellers = load_labellers(args.labellers_file)
    labeller_langs = set(
        lang
        for info in labellers.values()
        for lang in info['language']
    )

    # Sample questions according to bins
    questions = sample_questions(
        args.input_files,
        args.samples_per_bin,
        labeller_langs
    )

    # Distribute sampled questions to labellers
    workloads, counts = distribute(
        questions,
        labellers,
        args.labels_per_question
    )

    # TODO: quickly assert that all questions assigned to a person he speaks the language

    # Write out JSONL workloads and the manifest
    write_workloads(workloads, args.output_dir)
    write_manifest(workloads, args.output_dir)
    print(f"Manifest written to {os.path.join(args.output_dir,'manifest.json')}")

    # Print assignment summary
    total_q = len(questions)
    total_assign = sum(counts.values())
    print(f"\nSampled questions: {total_q}")
    print(f"Total label assignments: {total_assign}")
    print("Per-labeller assignment counts:")
    for u, c in counts.items():
        print(f"  {u}: {c}")


if __name__ == "__main__":
    main()
