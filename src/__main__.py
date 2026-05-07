# ABOUTME: Main entry point for the function calling project.
# ABOUTME: Handles CLI arguments, loads input files, runs inference,
# ABOUTME: and writes output results.

import argparse
import json
import os
import sys
import time
from typing import Any

from src.models import FunctionDefinition, TestPrompt, FunctionCallResult
from src.vocabulary import Vocabulary
from src.constrained_decoder import ConstrainedDecoder


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Function calling with constrained decoding for LLMs"
    )
    parser.add_argument(
        "--functions_definition",
        type=str,
        default="data/input/functions_definition.json",
        help="Path to the function definitions JSON file",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/input/function_calling_tests.json",
        help="Path to the input prompts JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/output/function_calling_results.json",
        help="Path to the output results JSON file",
    )
    return parser.parse_args()


def load_functions(path: str) -> list[FunctionDefinition]:
    """Load and validate function definitions from a JSON file.

    Args:
        path: Path to the functions_definition.json file.

    Returns:
        List of validated FunctionDefinition objects.

    Raises:
        SystemExit: If the file is missing or contains invalid JSON.
    """
    if not os.path.exists(path):
        print(f"Error: Function definitions file not found: {path}")
        sys.exit(1)

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in function definitions file: {e}")
        sys.exit(1)

    if not isinstance(raw_data, list):
        print("Error: Function definitions must be a JSON array")
        sys.exit(1)

    functions: list[FunctionDefinition] = []
    for item in raw_data:
        try:
            fn = FunctionDefinition(**item)
            functions.append(fn)
        except Exception as e:
            print(f"Warning: Skipping invalid function definition: {e}")

    if not functions:
        print("Error: No valid function definitions found")
        sys.exit(1)

    return functions


def load_prompts(path: str) -> list[TestPrompt]:
    """Load and validate test prompts from a JSON file.

    Args:
        path: Path to the function_calling_tests.json file.

    Returns:
        List of validated TestPrompt objects.

    Raises:
        SystemExit: If the file is missing or contains invalid JSON.
    """
    if not os.path.exists(path):
        print(f"Error: Input file not found: {path}")
        sys.exit(1)

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in input file: {e}")
        sys.exit(1)

    if not isinstance(raw_data, list):
        print("Error: Input file must be a JSON array")
        sys.exit(1)

    prompts: list[TestPrompt] = []
    for item in raw_data:
        try:
            prompt = TestPrompt(**item)
            prompts.append(prompt)
        except Exception as e:
            print(f"Warning: Skipping invalid prompt: {e}")

    return prompts


def save_results(results: list[dict[str, Any]], path: str) -> None:
    """Save function call results to a JSON file.

    Creates the output directory if it doesn't exist.

    Args:
        results: List of result dictionaries.
        path: Output file path.
    """
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Results saved to: {path}")


def main() -> None:
    """Main entry point: load data, run inference, save results."""
    args = parse_args()

    print("=" * 60)
    print("  Call Me Maybe - Function Calling with Constrained Decoding")
    print("=" * 60)

    # Load input files
    print("\n[1/4] Loading function definitions...")
    functions = load_functions(args.functions_definition)
    print(f"  Loaded {len(functions)} function(s):")
    for fn in functions:
        params = ", ".join(
            f"{k}: {v.type}" for k, v in fn.parameters.items()
        )
        print(f"    - {fn.name}({params})")

    print("\n[2/4] Loading test prompts...")
    prompts = load_prompts(args.input)
    print(f"  Loaded {len(prompts)} prompt(s)")

    # Initialize the LLM and decoder
    print("\n[3/4] Initializing LLM model (Qwen/Qwen3-0.6B)...")
    try:
        from llm_sdk import Small_LLM_Model
        model = Small_LLM_Model()
    except Exception as e:
        print(f"Error: Failed to initialize LLM model: {e}")
        sys.exit(1)

    print("  Loading vocabulary...")
    try:
        tokenizer_path = model.get_path_to_tokenizer_file()
        vocab = Vocabulary(tokenizer_path)
        print(f"  Vocabulary size: {vocab.size} tokens")
    except Exception as e:
        print(f"Error: Failed to load vocabulary: {e}")
        sys.exit(1)

    decoder = ConstrainedDecoder(model, vocab)

    # Process all prompts
    print(f"\n[4/4] Processing {len(prompts)} prompt(s)...")
    results: list[dict[str, Any]] = []
    start_time = time.time()

    for i, test_prompt in enumerate(prompts):
        prompt_start = time.time()
        print(f"\n  [{i + 1}/{len(prompts)}] \"{test_prompt.prompt}\"")

        try:
            result = decoder.process_prompt(functions, test_prompt.prompt)
            # Validate with pydantic
            validated = FunctionCallResult(**result)
            results.append(validated.model_dump())
            elapsed = time.time() - prompt_start
            print(
                f"    -> {result['name']}({result['parameters']}) "
                f"[{elapsed:.1f}s]"
            )
        except Exception as e:
            print(f"    Error processing prompt: {e}")
            # Add a best-effort result
            results.append({
                "prompt": test_prompt.prompt,
                "name": functions[0].name,
                "parameters": {
                    k: 0.0 if v.type == "number" else ""
                    for k, v in functions[0].parameters.items()
                },
            })

    total_time = time.time() - start_time
    print(f"\n  Total processing time: {total_time:.1f}s")

    # Save results
    print("\nSaving results...")
    save_results(results, args.output)

    print("\nDone!")


if __name__ == "__main__":
    main()
