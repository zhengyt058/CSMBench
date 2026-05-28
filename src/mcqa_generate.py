#!/usr/bin/env python3
"""Run multiple-choice caption matching (MCQA) evaluation on CSMBench."""

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from openai import OpenAI

from utils import (
    build_mcqa_prompt,
    get_client_config,
    get_repo_root,
    image_to_data_url,
    resolve_figure_path,
)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None, **kwargs):
        if desc:
            print(f"{desc}...")
        return iterable


DEFAULT_MODEL = "gpt-4o"


def process_single_entry(entry, model, client, data_root, logger=None):
    index = entry.get("index")
    answer_gt = str(entry.get("correct_answer", "")).strip().upper()
    scale = entry.get("scale", "Unknown")
    figure_path = entry.get("figure_path", "")

    full_image_path = resolve_figure_path(figure_path, data_root)
    if not full_image_path.is_file():
        error_msg = f"Image not found: {full_image_path}"
        if logger:
            logger.error(f"[Index {index}] {error_msg}")
        return {
            "index": index,
            "scale": scale,
            "pred": "Error",
            "correct": answer_gt,
            "is_correct": False,
            "output": error_msg,
        }

    try:
        data_url = image_to_data_url(full_image_path)
    except Exception as e:
        error_msg = f"Encode error: {e}"
        if logger:
            logger.error(f"[Index {index}] {error_msg}")
        return {
            "index": index,
            "scale": scale,
            "pred": "Error",
            "correct": answer_gt,
            "is_correct": False,
            "output": error_msg,
        }

    options = entry.get("options", [])
    prompt = build_mcqa_prompt(options)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }
    ]

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=64,
            timeout=120,
        )
        raw = resp.choices[0].message.content.strip()
        pred = None
        for ch in raw:
            if ch.upper() in ["A", "B", "C", "D"]:
                pred = ch.upper()
                break
        if pred is None:
            pred = raw[:1].upper() if raw else "?"
        is_correct = pred == answer_gt

        if logger:
            status = "✓" if is_correct else "✗"
            logger.info(f"[Index {index}] [{scale}] {status} Pred={pred} GT={answer_gt}")

        return {
            "index": index,
            "scale": scale,
            "pred": pred,
            "correct": answer_gt,
            "is_correct": is_correct,
            "output": raw,
        }
    except Exception as e:
        error_msg = f"Inference error: {e}"
        if logger:
            logger.error(f"[Index {index}] {error_msg}")
        return {
            "index": index,
            "scale": scale,
            "pred": "Error",
            "correct": answer_gt,
            "is_correct": False,
            "output": error_msg,
        }


def setup_logger(log_file=None):
    logger = logging.getLogger("mcqa_generate")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def process_batch(input_file, model, output_dir, data_root, max_workers=10, log_file=None):
    logger = setup_logger(log_file)
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Found {len(data)} entries | model={model} | workers={max_workers}")
    logger.info(f"Processing {len(data)} MCQA entries with {model}")

    def _run(entry):
        client = OpenAI(**get_client_config())
        return process_single_entry(entry, model, client, data_root, logger=logger)

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_run, entry): entry for entry in data}
        for future in tqdm(as_completed(futures), total=len(data), desc=f"MCQA ({model})"):
            try:
                results.append(future.result())
            except Exception as e:
                entry = futures[future]
                results.append({
                    "index": entry.get("index"),
                    "scale": entry.get("scale", "Unknown"),
                    "pred": "Error",
                    "correct": entry.get("correct_answer"),
                    "is_correct": False,
                    "output": f"Task error: {e}",
                })

    results.sort(key=lambda x: x["index"])
    total = len(results)
    correct = sum(1 for r in results if r.get("is_correct"))
    accuracy = correct / total if total else 0.0

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model.replace('/', '_')}_mcqa_output.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"accuracy": accuracy, "results": results}, f, ensure_ascii=False, indent=2)

    print(f"\nAccuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"Output: {output_path}")


def main():
    repo_root = get_repo_root()
    parser = argparse.ArgumentParser(description="CSMBench MCQA inference")
    parser.add_argument(
        "--input",
        type=str,
        default=str(repo_root / "CSMBench_Data" / "multi_scale_mcq.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(repo_root / "outputs" / "mcqa"),
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(repo_root),
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--max-workers", type=int, default=10)
    parser.add_argument("--log-file", type=str, default=None)
    args = parser.parse_args()

    if args.log_file is None:
        log_dir = Path(args.output_dir) / "logs"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.log_file = str(log_dir / f"{args.model}_{timestamp}.log")

    process_batch(
        input_file=args.input,
        model=args.model,
        output_dir=args.output_dir,
        data_root=Path(args.data_root),
        max_workers=args.max_workers,
        log_file=args.log_file,
    )


if __name__ == "__main__":
    main()
