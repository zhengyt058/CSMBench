#!/usr/bin/env python3
"""Score OpenQA predictions with an LLM judge (0–10 scale)."""

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

from utils import create_openqa_prompt, get_client_config, get_repo_root

DEFAULT_JUDGE_MODEL = "gpt-4o"


def create_evaluation_prompt(question, ground_truth, prediction):
    return f"""
You are a strict evaluator assessing answer correctness. Score the model's prediction from 0 to 10.

Question: {question}
Ground Truth Answer: {ground_truth}
Model Prediction: {prediction}

Rules:
- High score if the prediction matches semantically; ignore minor formatting differences.
- Deduct points for partial correctness or incorrect extra information.

Return a single integer from 0 to 10 only.
"""


def extract_score(response_text):
    response_text = response_text.strip()
    if re.search(r"\b10\b", response_text):
        return 10
    match = re.search(r"\b([0-9])\b", response_text)
    if match:
        return min(max(int(match.group(1)), 0), 10)
    try:
        return min(max(int(response_text), 0), 10)
    except ValueError:
        return None


def score_single_entry(client, question, ground_truth, prediction, index, judge_model):
    messages = [{"role": "user", "content": create_evaluation_prompt(question, ground_truth, prediction)}]
    try:
        response = client.chat.completions.create(
            model=judge_model,
            messages=messages,
            max_tokens=64,
            timeout=60,
        )
        raw = response.choices[0].message.content.strip()
        return {"score": extract_score(raw), "raw_response": raw}
    except Exception as e:
        print(f"Error scoring entry {index}: {e}")
        return {"score": None, "raw_response": f"Error: {e}"}


def process_single_entry(args_tuple):
    entry, ref_dict, client_config, judge_model = args_tuple
    index = entry.get("index")
    prediction = entry.get("output", "")

    if not prediction or not str(prediction).strip():
        return {"index": index, "scale": entry.get("scale", ""), "score": None, "error": "Empty prediction"}

    if index not in ref_dict:
        return {"index": index, "scale": entry.get("scale", ""), "score": None, "error": "No reference found"}

    ref = ref_dict[index]
    image_caption = ref.get("image_caption", "")
    ground_truth_raw = ref.get("describe_paragraph", "")
    if isinstance(ground_truth_raw, list):
        ground_truth = " ".join(str(p) for p in ground_truth_raw)
    else:
        ground_truth = str(ground_truth_raw) if ground_truth_raw else ""

    if not ground_truth.strip():
        return {"index": index, "scale": entry.get("scale", ""), "score": None, "error": "Empty ground truth"}

    question = create_openqa_prompt(image_caption)
    client = OpenAI(**client_config)
    result = score_single_entry(client, question, ground_truth, prediction, index, judge_model)

    return {
        "index": index,
        "scale": entry.get("scale", ""),
        "score": result["score"],
        "raw_response": result["raw_response"],
        "error": None,
    }


def main():
    repo_root = get_repo_root()
    parser = argparse.ArgumentParser(description="Score CSMBench OpenQA outputs with LLM judge")
    parser.add_argument("output_file", type=str, help="Model output JSON from openqa_generate.py")
    parser.add_argument(
        "--reference",
        type=str,
        default=str(repo_root / "CSMBench_Data" / "multi_scale.json"),
    )
    parser.add_argument(
        "--scores-output",
        type=str,
        default=None,
        help="Output scores JSON path (default: outputs/scores/<stem>_scores.json)",
    )
    parser.add_argument("--judge-model", type=str, default=os.environ.get("JUDGE_MODEL", DEFAULT_JUDGE_MODEL))
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    if not os.path.exists(args.output_file):
        raise FileNotFoundError(f"Output file not found: {args.output_file}")
    if not os.path.exists(args.reference):
        raise FileNotFoundError(f"Reference file not found: {args.reference}")

    if args.scores_output:
        scores_path = Path(args.scores_output)
    else:
        stem = Path(args.output_file).stem.replace("_openqa_output", "")
        scores_path = repo_root / "outputs" / "scores" / f"{stem}_openqa_scores.json"

    with open(args.output_file, "r", encoding="utf-8") as f:
        outputs = json.load(f)
    with open(args.reference, "r", encoding="utf-8") as f:
        references = json.load(f)

    ref_dict = {ref["index"]: ref for ref in references}
    client_config = get_client_config()
    tasks = [(entry, ref_dict, client_config, args.judge_model) for entry in outputs]

    scored_results = []
    failed_entries = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_entry, t): t[0] for t in tasks}
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Scoring"):
            result = future.result()
            if result.get("error") or result.get("score") is None:
                failed_entries.append(result)
            else:
                scored_results.append(result)

    scored_results.sort(key=lambda x: x["index"])
    valid_scores = [r["score"] for r in scored_results if r["score"] is not None]

    scale_stats = {}
    for r in scored_results:
        if r["score"] is None:
            continue
        scale = r.get("scale", "Unknown")
        scale_stats.setdefault(scale, []).append(r["score"])

    scale_averages = {
        scale: {
            "average_score": sum(scores) / len(scores),
            "count": len(scores),
        }
        for scale, scores in scale_stats.items()
    }

    stats = {
        "total_entries": len(outputs),
        "successfully_scored": len(valid_scores),
        "failed_entries": len(failed_entries),
        "average_score": sum(valid_scores) / len(valid_scores) if valid_scores else None,
        "by_scale": scale_averages,
    }

    scores_path.parent.mkdir(parents=True, exist_ok=True)
    with open(scores_path, "w", encoding="utf-8") as f:
        json.dump({"statistics": stats, "results": scored_results, "failed_entries": failed_entries}, f, indent=2)

    print(f"Scored: {stats['successfully_scored']}/{stats['total_entries']}")
    if stats["average_score"] is not None:
        print(f"Average score: {stats['average_score']:.2f}")
    print(f"Saved to: {scores_path}")


if __name__ == "__main__":
    main()
