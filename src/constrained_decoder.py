# ABOUTME: Core constrained decoding engine for JSON generation.
# ABOUTME: Guarantees 100% valid, schema-compliant JSON output by masking
# ABOUTME: invalid tokens at each generation step.

from typing import Any

from llm_sdk import Small_LLM_Model
from src.models import FunctionDefinition
from src.vocabulary import Vocabulary
from src.prompt_builder import build_function_selection_prompt


class ConstrainedDecoder:
    """Generates structured function calls using constrained decoding."""

    def __init__(self, model: Small_LLM_Model, vocab: Vocabulary) -> None:
        """Initialize the decoder with the model and vocabulary.

        Args:
            model: The LLM model instance.
            vocab: The loaded vocabulary object.
        """
        self.model = model

        # Normalize special characters: Ġ = space, Ċ = newline
        self._tokens: dict[int, str] = {}
        for i, t in vocab.id_to_token.items():
            norm = t.replace("Ġ", " ").replace("Ċ", "\n").replace("ĉ", "\t")
            if norm:
                self._tokens[i] = norm

        # Pre-filter token subsets once at startup for speed
        # Number tokens: only contain digits, dot, or minus
        self._num_tokens: dict[int, str] = {
            i: t for i, t in self._tokens.items()
            if all(c in "0123456789.-" for c in t)
        }

        # String tokens: no quotes, only printable characters
        self._str_tokens: dict[int, str] = {
            i: t for i, t in self._tokens.items()
            if '"' not in t and all(ord(c) >= 32 for c in t)
        }

        # The token ID for a double quote character
        self._quote_id: int | None = next(
            (i for i, t in self._tokens.items() if t == '"'), None
        )

    def process_prompt(
        self, functions: list[FunctionDefinition], user_prompt: str
    ) -> dict[str, Any]:
        """Turn a user prompt into a structured function call dictionary.

        Args:
            functions: List of available function definitions.
            user_prompt: The natural language request from the user.

        Returns:
            A dictionary with keys: prompt, name, parameters.
        """
        prompt = build_function_selection_prompt(functions, user_prompt)

        # Start building the output JSON prefix
        prefix = '{"name": "'
        input_ids = self.model.encode(prompt + prefix)[0].tolist()

        # Step 1: let the model pick the function name
        fn_name = self._generate_options(input_ids, [f.name for f in functions])
        fn = next((f for f in functions if f.name == fn_name), functions[0])

        # Step 2: generate each argument based on its type
        args: dict[str, Any] = {}
        prefix += fn.name + '", "parameters": {'

        for i, (p_name, p_def) in enumerate(fn.parameters.items()):
            sep = ", " if i > 0 else ""
            prefix += sep + f'"{p_name}": '
            ids = self.model.encode(prompt + prefix)[0].tolist()

            if p_def.type == "boolean":
                val_str = self._generate_options(ids, ["true", "false"])
                args[p_name] = val_str == "true"
                prefix += val_str

            elif p_def.type == "number":
                val_str = self._generate_number(ids)
                try:
                    args[p_name] = float(val_str)
                except ValueError:
                    args[p_name] = 0.0
                prefix += val_str

            else:  # string
                val_str = self._generate_string(ids)
                args[p_name] = val_str
                prefix += '"' + val_str.replace('\\', '\\\\').replace('"', '\\"') + '"'

        return {
            "prompt": user_prompt,
            "name": fn.name,
            "parameters": args,
        }

    def _generate_options(
        self, input_ids: list[int], options: list[str]
    ) -> str:
        """Pick exactly one option by generating tokens one at a time.

        Args:
            input_ids: Current token ID sequence fed to the model.
            options: List of valid string options to choose from.

        Returns:
            The selected option string.
        """
        generated = ""

        while True:
            # Which options still match what we have built so far?
            still_valid = [o for o in options if o.startswith(generated)]

            # Done: only one option left, or nothing matched (use fallback)
            if len(still_valid) <= 1:
                return still_valid[0] if still_valid else options[0]

            # Ask the model: which token comes next?
            logits = self.model.get_logits_from_input_ids(input_ids)

            # Find the highest-scoring token that continues a valid option
            best_id: int | None = None
            best_score = -float("inf")

            for token_id, token_str in self._tokens.items():
                candidate = generated + token_str
                if any(o.startswith(candidate) for o in still_valid):
                    if logits[token_id] > best_score:
                        best_score, best_id = logits[token_id], token_id

            # No valid token found — return best remaining option
            if best_id is None:
                return still_valid[0]

            generated += self._tokens[best_id]
            input_ids.append(best_id)

            # Exact match — we are done
            if generated in options:
                return generated

    def _generate_number(self, input_ids: list[int]) -> str:
        """Generate a valid number string like 40.0 or -3.14.

        Args:
            input_ids: Current token ID sequence fed to the model.

        Returns:
            A string representing a valid float number.
        """
        generated = ""

        for _ in range(20):  # max 20 tokens is more than enough for any number
            logits = self.model.get_logits_from_input_ids(input_ids)

            best_id: int | None = None
            best_score = -float("inf")

            for token_id, token_str in self._num_tokens.items():
                candidate = generated + token_str
                if self._is_valid_num_prefix(candidate):
                    if logits[token_id] > best_score:
                        best_score, best_id = logits[token_id], token_id

            # No valid number token scored higher — the number is complete
            if best_id is None:
                break

            generated += self._tokens[best_id]
            input_ids.append(best_id)

        return generated if self._is_valid_num(generated) else "0.0"

    def _generate_string(self, input_ids: list[int]) -> str:
        """Generate a string value without surrounding quotes.

        Args:
            input_ids: Current token ID sequence fed to the model.

        Returns:
            The string content (no quotes).
        """
        generated = ""

        # Add opening quote to the context so the model knows we are in a string
        if self._quote_id is not None:
            input_ids.append(self._quote_id)

        for _ in range(50):  # max 50 tokens for a string value
            logits = self.model.get_logits_from_input_ids(input_ids)

            # Start by considering the closing quote as a candidate
            best_id: int | None = self._quote_id
            best_score = (
                logits[self._quote_id]
                if self._quote_id is not None
                else -float("inf")
            )

            # Consider all safe printable string tokens
            for token_id, token_str in self._str_tokens.items():
                if logits[token_id] > best_score:
                    best_score, best_id = logits[token_id], token_id

            # Model chose to close the string — we are done
            if best_id == self._quote_id:
                return generated

            if best_id is None:
                return generated

            generated += self._tokens[best_id]
            input_ids.append(best_id)

        return generated

    def _is_valid_num_prefix(self, s: str) -> bool:
        """Check if s could be the beginning of a valid number.

        Args:
            s: The string to check.

        Returns:
            True if s is a valid number prefix.
        """
        # A lone minus or empty string is a valid start
        if s in ("", "-"):
            return True
        # A number ending with a dot like "3." is a valid prefix
        if s.endswith("."):
            try:
                float(s[:-1])
                return True
            except ValueError:
                return False
        # Otherwise it must already be a valid float
        return self._is_valid_num(s)

    def _is_valid_num(self, s: str) -> bool:
        """Check if s is a complete valid number.

        Args:
            s: The string to check.

        Returns:
            True if s can be parsed as a float.
        """
        try:
            float(s)
            return True
        except ValueError:
            return False