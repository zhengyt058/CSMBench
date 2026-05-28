#!/usr/bin/env python3
"""Run open-ended figure description (OpenQA) evaluation on CSMBench."""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

from utils import (
    create_openqa_prompt,
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


def process_single_entry(entry, model, client, data_root):
    index = entry.get("index")
    figure_path = entry.get("figure_path")
    image_caption = entry.get("image_caption", "")

    full_image_path = resolve_figure_path(figure_path, data_root)
    if not full_image_path.is_file():
        return {"index": index, "scale": entry.get("scale", ""), "output": f"Error: Image not found at {full_image_path}"}

    try:
        data_url = image_to_data_url(full_image_path)
    except Exception as e:
        return {"index": index, "scale": entry.get("scale", ""), "output": f"Error encoding image: {e}"}

    prompt = create_openqa_prompt(image_caption)
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
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=8000,
            timeout=120,
        )
        output = response.choices[0].message.content.strip()
        return {"index": index, "scale": entry.get("scale", ""), "output": output}
    except Exception as e:
        return {"index": index, "scale": entry.get("scale", ""), "output": f"Error during inference: {e}"}


def process_single_entry_with_client(entry, model, data_root):
    client = OpenAI(**get_client_config())
    return process_single_entry(entry, model, client, data_root)


def process_batch(input_file, model, output_dir, data_root, max_workers=10):
    print(f"Loading input file: {input_file}")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Found {len(data)} entries | model={model} | workers={max_workers}")

    results = []
    failed_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_entry_with_client, entry, model, data_root): entry
            for entry in data
        }
        for future in tqdm(as_completed(futures), total=len(data), desc=f"OpenQA ({model})"):
            try:
                result = future.result()
                results.append(result)
                if str(result.get("output", "")).startswith("Error"):
                    failed_count += 1
            except Exception as e:
                entry = futures[future]
                results.append({
                    "index": entry.get("index"),
                    "scale": entry.get("scale", ""),
                    "output": f"Error during task execution: {e}",
                })
                failed_count += 1

    results.sort(key=lambda x: x["index"])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model.replace('/', '_')}_openqa_output.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Success: {len(results) - failed_count}/{len(results)}")
    print(f"Output: {output_path}")
    return results


def main():
    repo_root = get_repo_root()
    parser = argparse.ArgumentParser(description="CSMBench OpenQA inference")
    parser.add_argument(
        "--input",
        type=str,
        default=str(repo_root / "CSMBench_Data" / "multi_scale.json"),
        help="Input JSON (open QA subset)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(repo_root / "outputs" / "openqa"),
        help="Directory for model outputs",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(repo_root),
        help="Root for resolving relative figure_path entries",
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--max-workers", type=int, default=10)
    args = parser.parse_args()

    process_batch(
        input_file=args.input,
        model=args.model,
        output_dir=args.output_dir,
        data_root=Path(args.data_root),
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()
