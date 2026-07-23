import json
from typing import Optional


class Vocabulary:
    def __init__(self, tokenizer_path: str) -> None:
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}
        self.load_vocabulary(tokenizer_path)

    # create the method that load the voca from the tokenzier
    def load_vocabulary(self, tokenizer_path: str) -> None:
        with open(tokenizer_path, "r", encoding="utf-8") as f:
            # now we need to load the tokenizer from path
            # convert json to python dict object
            tokenizer_data = json.load(f)
        model_data = tokenizer_data.get("model", {})
        vocab: dict[str, int] = model_data.get("vocab", {})
        # this is a fall back if vocab is empty
        if not vocab:
            added_tokens = tokenizer_data.get("added_tokens", [])
            for entry in added_tokens:
                entry_id = entry.get("id")
                content = entry.get("content")
                # only add the ones with an id
                if entry_id is not None:
                    vocab[content] = entry_id

        self.token_to_id = vocab
        self.id_to_token = {v: k for k, v in vocab.items()}

    # get the token id for the token str
    def get_token_id(self, token_str: str) -> Optional[int]:
        return self.token_to_id.get(token_str)

    # get the token str for the token id
    def get_token_str(self, token_id: int) -> Optional[str]:
        return self.id_to_token.get(token_id)

    # get all token ids
    def get_token_ids(self) -> list[int]:
        return list(self.id_to_token.keys())
