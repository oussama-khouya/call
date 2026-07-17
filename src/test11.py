from llm_sdk import Small_LLM_Model

model = Small_LLM_Model()

vocab_path = model.get_path_to_tokenizer_file()
print("Tokenizer file:", vocab_path)

with open(vocab_path, "r", encoding="utf-8") as f:
    print(f.read())