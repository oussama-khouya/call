import json 
from pathlib import Path
from llm_sdk import Small_LLM_Model
from .vocabulary import Vocabulary
from .constrained_decoder import constrained_decoding
from .models import functiondef , prompt , FunctionCallResult
import argparse


# create file path to make things easy 
# this is a hardcoded way we have to make command line interface methode by using argparse library 

#create a parser obj from a class named ARrgumentparser()
parser_00 = argparse.ArgumentParser(description="maybe_one_day_constained decoding")
parser_00.add_argument("--functions_definition", type=str, default="data/input/functions_definition.json", help="Path to the functions definition JSON file")
parser_00.add_argument("--input", type=str, default="data/input/function_calling_tests.json", help="path to the function calling tests")
parser_00.add_argument("--output", type=str, default="data/output/function_calling_results.json")

args = parser_00.parse_args()




def main():

    # load the functions from json  to python object dict using json.load(f) then to a pydantic object so we can validate its data automaticlly
    try:
        with open(args.functions_definition, "r", encoding="utf-8") as f:
                # For each dictionary in that list, model_validate converts it from a raw dict into a proper functiondef pydantic object
            functions = [functiondef.model_validate(fun) for fun in json.load(f)]
    except FileNotFoundError as e:
        print(f"functions_definition file does not found: {e}")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Input prompts file contains invalid JSON: {e}")
        return
    except Exception as e:
        print(f"error in the functions definition file as {e}")
        return


    # load the prompts 
    try:
        with open (args.input, "r", encoding="utf-8") as f:
            prompts = [prompt.model_validate(prom) for prom in json.load(f)]
    except FileNotFoundError as e:
        print(f"Error: Input prompts file not: {e}")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Input prompts file contains invalid JSON: {e}")
        return
    except Exception as e:
        print(f"Error reading input prompts: {e}")
        return



    # load the model and vocab
    print("loading the model...")
    try:
        model = Small_LLM_Model()
        vocab =  Vocabulary(model.get_path_to_tokenizer_file())
        decoder  = constrained_decoding(model, vocab)
    except Exception as e:
        print(f"Error initializing model or vocabulary: {e}")
        return

    # process each prompt

    results = []
    for i , p in enumerate(prompts, 1):
        print(f'[{i} / {len(prompts)}] / "{p.prompt}"')
        try:
            result : FunctionCallResult =  decoder.process_prompt(functions, p.prompt)
            results.append(result)
            #for display
            print(f"  -> {result.name}({result.parameters})")
        except Exception as e:
            print(f"-> Error processing prompt: {e}")
            continue
        


    # SAVE THE RESULTS to output file
    # we gonna use path for more dianamic use 

    out = Path(args.output)
    # then we should create that data/output folder to write in it 
    # .parent this create the parent folder 
    out.parent.mkdir(parents=True, exist_ok=True)
    # we open and write into the file
    try:
        with open(out, "w", encoding="utf-8") as f:
            json.dump([r.model_dump() for r in results],f, indent=2, ensure_ascii=True )
        print(f"\nDone. {len(results)} result(s) written to {args.output}")
    except Exception as e:
        print(f"Error writing to output file: {e}")
        return


if __name__ == "__main__":
    main()






