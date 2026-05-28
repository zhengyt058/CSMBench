"""Shared utilities for CSMBench evaluation."""

import base64
import os
from pathlib import Path


def get_repo_root() -> Path:
    """Return CSMBench repository root (parent of src/)."""
    return Path(__file__).resolve().parent.parent


def get_client_config() -> dict:
    """Build OpenAI-compatible client config from environment variables."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Export it before running evaluation, e.g.\n"
            "  export OPENAI_API_KEY='your-key'"
        )
    config = {"api_key": api_key}
    base_url = os.environ.get("OPENAI_BASE_URL")
    if base_url:
        config["base_url"] = base_url
    return config


def resolve_figure_path(figure_path: str, data_root: Path) -> Path:
    """Resolve figure path relative to data_root or as absolute path."""
    path = Path(figure_path)
    if path.is_absolute():
        return path
    return data_root / figure_path


def image_to_data_url(image_path) -> str:
    """Encode a local image as a base64 data URL for vision APIs."""
    image_path = str(image_path)
    lower = image_path.lower()
    if lower.endswith(".png"):
        mime_type = "image/png"
    elif lower.endswith((".jpg", ".jpeg")):
        mime_type = "image/jpeg"
    elif lower.endswith(".gif"):
        mime_type = "image/gif"
    elif lower.endswith(".webp"):
        mime_type = "image/webp"
    else:
        raise ValueError("Unsupported image format. Use PNG, JPG, GIF, or WebP.")

    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def create_openqa_prompt(caption: str) -> str:
    """Prompt for open-ended figure description (Task 1)."""
    return (
        "You are a helpful materials science assistant. Below is a morphological "
        "analysis figure extracted from materials literature. The caption of the "
        f"figure is: {caption}. Please provide a detailed analysis and explanation "
        "of the content depicted in the image. Focus on the visible features of the "
        "image and draw conclusions based on what is presented. Note: There is a "
        "word count limit for your response. Your explanation should be between "
        "100 and 300 words."
    )


def build_mcqa_prompt(options: list) -> str:
    """Prompt for multiple-choice caption matching (Task 2)."""
    options_text = "\n".join(f"{opt['label']}. {opt['caption']}" for opt in options)
    return (
        "You are given an image and four candidate captions. Exactly one caption "
        "correctly describes the image.\n"
        "Return ONLY the letter of the correct caption (A, B, C, or D). "
        "Do not return any other text.\n\n"
        "Example response: C\n\n"
        f"Options:\n{options_text}\n\n"
        "IMPORTANT: Return only the letter (A, B, C, or D). No explanation."
    )
