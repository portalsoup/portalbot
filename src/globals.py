import argparse
import json
from typing import Any

import torch

def init_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI Thrallbot")

    parser.add_argument("-p", "--prompt", type=str, help="Path to prompt file")
    parser.add_argument("--no-cuda", type=bool, default=False, help="Disable CUDA")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("-m", "--model", type=str, default="Qwen/Qwen3-1.7B", help="Model to use")
    parser.add_argument("-t", "--think", action="store_true", help="Enable thinking")

    return parser.parse_args()

def handle_if_quit(prompt):
    if prompt.lower() in {"exit", "quit"}:
        print("Exiting...")
        exit(0)


def load_prompt(path: str) -> Any:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, IndexError):
        print("Usage: python main.py <prompt.json>")
        exit(1)


def init_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    else:
        print("CUDA is not available.")

