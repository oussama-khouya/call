import argparse
import json
from pathlib import Path
from typing import Optional

from llm_sdk import Small_LLM_Model

from .constrained_decoder import constrained_decoding
from .models import FunctionCallResult, functiondef, prompt
from .vocabulary import Vocabulary


parser_00 = argparse.ArgumentParser(
    description="constrained decoding for function calling"
)
parser_00.add_argument(
    "--functions_definition",
    type=str,
    default="data/input/functions_definition.json",
    help="Path to the functions definition JSON file",
)
parser_00.add_argument(
    "--input",
    type=str,
    default="data/input/function_calling_tests.json",
    help="Path to the function calling tests",
)
parser_00.add_argument(
    "--output",
    type=str,
    default="data/output/function_calling_results.json",
    help="Path to the output JSON file",
)
# parse the input 
args = parser_00.parse_args()


def main() -> None:
    # Load functions from JSON to pydantic functiondef objects
    try:
        with open(args.functions_definition, "r", encoding="utf-8") as f:
            functions = [
                functiondef.model_validate(fun) for fun in json.load(f)
            ]
    except FileNotFoundError as e:
        print(f"functions_definition file not found: {e}")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Input prompts file contains invalid JSON: {e}")
        return
    except Exception as e:
        print(f"Error in functions definition file: {e}")
        return

    # Load prompts
    try:
        with open(args.input, "r", encoding="utf-8") as f:
            prompts = [prompt.model_validate(prom) for prom in json.load(f)]
    except FileNotFoundError as e:
        print(f"Error: Input prompts file not found: {e}")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Input prompts file contains invalid JSON: {e}")
        return
    except Exception as e:
        print(f"Error reading input prompts: {e}")
        return

    # Load model and vocabulary
    print("Loading the model...")
    try:
        model = Small_LLM_Model()
        vocab = Vocabulary(model.get_path_to_tokenizer_file())
        decoder = constrained_decoding(model, vocab)
    except Exception as e:
        print(f"Error initializing model or vocabulary: {e}")
        return

    # Process each prompt
    results: list[FunctionCallResult] = []
    for i, p in enumerate(prompts, 1):
        print(f'[{i} / {len(prompts)}] / "{p.prompt}"')
        try:
            result: Optional[FunctionCallResult] = decoder.process_prompt(
                functions, p.prompt
            )
            if result is None:
                return
            results.append(result)
            print(f"  -> {result.name}({result.parameters})")
        except Exception as e:
            print(f"-> Error processing prompt: {e}")
            continue

    # Save results to output file
    # model_dump from pydantic to dict
    # jsondump write that into f as json 
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(out, "w", encoding="utf-8") as f:
            json.dump(
                [r.model_dump() for r in results],
                f,
                indent=2,
            )
        print(f"\nDone. {len(results)} result(s) written to {args.output}")
    except Exception as e:
        print(f"Error writing to output file: {e}")
        return


if __name__ == "__main__":
    main()
