#!/usr/bin/env python3
"""
Convert Hugging Face ImageFolder download to JSON files for evaluation.

After downloading the dataset:
  huggingface-cli download --repo-type dataset lututu/CSMBench --local-dir CSMBench_Data

Run:
  python prepare_dataset.py --data-dir ../CSMBench_Data
"""

import argparse
import json
from pathlib import Path

from utils import get_repo_root


def load_metadata_jsonl(path: Path) -> list:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def convert_config(data_dir: Path, config_name: str, is_mcq: bool) -> list:
    """Convert one HF config (multi_scale or multi_scale_mcq) to eval JSON."""
    train_dir = data_dir / config_name / "train"
    metadata_path = train_dir / "metadata.jsonl"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata: {metadata_path}")

    entries = []
    for row in load_metadata_jsonl(metadata_path):
        file_name = row["file_name"]
        figure_path = str(train_dir / file_name)

        entry = {
            "index": row.get("index", len(entries)),
            "scale": row.get("scale", ""),
            "source": row.get("source", ""),
            "paper_folder_name": row.get("paper_folder_name", ""),
            "figure_path": figure_path,
        }
        if row.get("hybrid") is not None:
            entry["hybrid"] = row["hybrid"]

        if is_mcq:
            options = row.get("options", "[]")
            if isinstance(options, str):
                options = json.loads(options)
            entry["options"] = options
            entry["correct_answer"] = row.get("correct_answer", "")
        else:
            entry["image_caption"] = row.get("image_caption", "")
            entry["describe_paragraph"] = row.get("describe_paragraph", "")

        entries.append(entry)

    entries.sort(key=lambda x: x["index"])
    return entries


def main():
    repo_root = get_repo_root()
    parser = argparse.ArgumentParser(description="Prepare CSMBench JSON from HF download")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(repo_root / "CSMBench_Data"),
        help="Directory from huggingface-cli download",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(repo_root / "CSMBench_Data"),
        help="Where to write multi_scale.json and multi_scale_mcq.json",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    openqa = convert_config(data_dir, "multi_scale", is_mcq=False)
    mcqa = convert_config(data_dir, "multi_scale_mcq", is_mcq=True)

    openqa_path = output_dir / "multi_scale.json"
    mcqa_path = output_dir / "multi_scale_mcq.json"

    openqa_path.write_text(json.dumps(openqa, ensure_ascii=False, indent=2), encoding="utf-8")
    mcqa_path.write_text(json.dumps(mcqa, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {len(openqa)} OpenQA entries -> {openqa_path}")
    print(f"Wrote {len(mcqa)} MCQA entries -> {mcqa_path}")


if __name__ == "__main__":
    main()
