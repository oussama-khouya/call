# ABOUTME: Vocabulary handling for the LLM tokenizer.
# ABOUTME: Loads the tokenizer vocabulary and builds mappings between
# ABOUTME: token IDs and their string representations for constrained decoding.

import json
from typing import Optional


class Vocabulary:
    """Manages the tokenizer vocabulary for constrained decoding.

    Loads the vocabulary from the tokenizer JSON file and provides
    efficient lookup between token strings and their IDs.

    Attributes:
        token_to_id: Mapping from token string to token ID.
        id_to_token: Mapping from token ID to token string.
    """

    def __init__(self, tokenizer_path: str) -> None:
        """Initialize vocabulary from a tokenizer JSON file.

        Args:
            tokenizer_path: Path to the tokenizer.json file.
        """
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}
        self._load_vocabulary(tokenizer_path)

    def _load_vocabulary(self, tokenizer_path: str) -> None:
        """Load and parse the tokenizer vocabulary file.

        Args:
            tokenizer_path: Path to the tokenizer.json file.
        """
        with open(tokenizer_path, "r", encoding="utf-8") as f:
            tokenizer_data = json.load(f)

        # The tokenizer.json has a "model" -> "vocab" mapping
        model_data = tokenizer_data.get("model", {})
        vocab: dict[str, int] = model_data.get("vocab", {})

        if not vocab:
            # Fallback: try the added_tokens or other structures
            added_tokens = tokenizer_data.get("added_tokens", [])
            for token_entry in added_tokens:
                tid = token_entry.get("id")
                content = token_entry.get("content", "")
                if tid is not None:
                    vocab[content] = tid

        self.token_to_id = vocab
        self.id_to_token = {v: k for k, v in vocab.items()}

    def get_token_string(self, token_id: int) -> Optional[str]:
        """Get the string representation of a token ID.

        Args:
            token_id: The numerical token ID.

        Returns:
            The string representation, or None if not found.
        """
        return self.id_to_token.get(token_id)

    def get_token_id(self, token_string: str) -> Optional[int]:
        """Get the token ID for a string.

        Args:
            token_string: The token string.

        Returns:
            The token ID, or None if not found.
        """
        return self.token_to_id.get(token_string)

    def get_all_token_ids(self) -> list[int]:
        """Return all token IDs in the vocabulary.

        Returns:
            List of all token IDs.
        """
        return list(self.id_to_token.keys())

    @property
    def size(self) -> int:
        """Return the total vocabulary size."""
        return len(self.id_to_token)
