import sys
sys.path.insert(0, "/Users/okhouya/Documents/call/llm_sdk")

from llm_sdk import Small_LLM_Model
import json

model = Small_LLM_Model()
path = model.get_path_to_tokenizer_file()

with open(path) as f:
    data = json.load(f)

vocab = data.get("model", {}).get("vocab", {})

# find minus related tokens
minus_tokens = {k: v for k, v in vocab.items() if "-" in k}
print("minus tokens:", list(minus_tokens.items())[:20])