#!/usr/bin/env python3
"""Compute MCQA accuracy overall and by physical scale."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from utils import get_repo_root


def calculate_statistics(results_file: Path, output_file: Path = None):
    with open(results_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "results" in data:
        results = data["results"]
        overall_accuracy = data.get("accuracy")
    else:
        results = data
        overall_accuracy = None

    scale_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    for result in results:
        scale = result.get("scale", "Unknown")
        scale_stats[scale]["total"] += 1
        if result.get("is_correct", False):
            scale_stats[scale]["correct"] += 1

    scale_accuracies = {}
    for scale, stats in scale_stats.items():
        if stats["total"] > 0:
            scale_accuracies[scale] = {
                "accuracy": stats["correct"] / stats["total"],
                "correct": stats["correct"],
                "total": stats["total"],
            }

    if overall_accuracy is None:
        total_correct = sum(1 for r in results if r.get("is_correct", False))
        overall_accuracy = total_correct / len(results) if results else 0.0

    print("=" * 60)
    print("MCQA Results Statistics")
    print("=" * 60)
    print(f"\nOverall Accuracy: {overall_accuracy:.4f} ({overall_accuracy * 100:.2f}%)")
    print(f"Total samples: {len(results)}")

    print("\nAccuracy by Scale:")
    print("-" * 60)
    scale_order = ["AtomScale", "Microscale", "Mesoscale", "Macroscale"]
    for scale in scale_order:
        if scale in scale_accuracies:
            s = scale_accuracies[scale]
            print(f"{scale:15s}: {s['accuracy']:.4f} ({s['accuracy']*100:.2f}%) [{s['correct']}/{s['total']}]")
    for scale, s in sorted(scale_accuracies.items()):
        if scale not in scale_order:
            print(f"{scale:15s}: {s['accuracy']:.4f} ({s['accuracy']*100:.2f}%) [{s['correct']}/{s['total']}]")
    print("=" * 60)

    summary = {
        "overall_accuracy": overall_accuracy,
        "total_samples": len(results),
        "by_scale": scale_accuracies,
    }

    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\nStatistics saved to: {output_file}")

    return summary


def main():
    repo_root = get_repo_root()
    parser = argparse.ArgumentParser(description="CSMBench MCQA statistics")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="MCQA output JSON from mcqa_generate.py",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save statistics JSON",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path = Path(args.output) if args.output else None
    calculate_statistics(input_path, output_path)


if __name__ == "__main__":
    main()
