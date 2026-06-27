import json 
from pathlib import Path
from llm_sdk import Small_LLM_Model
from .vocabulay import Vocabulary
from .constrained_decoder import constrained_decoding
from .models import functiondef , prompt , FunctionCallResult


# create file path to make things easy 

functions_file = "data/input/functions_definition.json"
prompt_file = "data/input/function_calling_tests.json"
output_file = "data/output/function_calling_results.json"



def main():

    # load the functions from json  to python object dict using json.load(f) then to a pydantic object so we can validate its data automaticlly
    with open(functions_file, "r", encoding="utf-8") as f:
            # For each dictionary in that list, model_validate converts it from a raw dict into a proper functiondef pydantic object
        functions = [functiondef.model_validate(fun) for fun in json.load(f)]


    # load the prompts 
    with open (prompt_file, "r", encoding="utf-8") as f:
        prompts = [prompt.model_validate(prom) for prom in json.load(f)]



    # load the model and vocab
    print("loading the model...")
    model = Small_LLM_Model()
    vocab =  Vocabulary(model.get_path_to_vocab_file())
    decoder  = constrained_decoding(model, vocab)

    # process each prompt

    results = []
    for i , p in enumerate(prompts, 1):
        print(f'[{i} / {len(prompts)}] / "{p.prompt}"')
        result : FunctionCallResult =  decoder.process_prompt(functions, p.prompt)
        results.append(result)
        #for display
        print(f"  -> {result.name}({result.parameters})")


    # SAVE THE RESULTS to output file
    # we gonna use path for more dianamic use 

    out = Path(output_file)
    # then we should create that data/output folder to write in it 
    # .parent this create the parent folder 
    out.parent.mkdir(parents=True, exist_ok=True)
    # we open and write into the file
    with open(out, "w", encoding="utf-8") as f:
        json.dump([r.model_dump() for r in results],f, indent=2, ensure_ascii=True )
    print(f"\nDone. {len(results)} result(s) written to {output_file}")


if __name__ == "__main__":
    main()






