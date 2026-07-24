from typing import Any, Optional

from llm_sdk import Small_LLM_Model

from src.models import FunctionCallResult, functiondef
from src.prompt_builder import prompt_builder
from src.vocabulary import Vocabulary


class constrained_decoding:
    def __init__(self, model: Small_LLM_Model, vocab: Vocabulary) -> None:
        self.model = model

        # filter and replace space
        self.tokens: dict[int, str] = {}
        for token_id, token in vocab.id_to_token.items():
            clean = (
                token.replace("Ġ", " ")
                .replace("Ċ", "\n")
                .replace("ĉ", "\t")
            )

            if clean:
                self.tokens[token_id] = clean

        # filter and save number tokens
        self.num_tokens: dict[int, str] = {
            i: token
            for i, token in self.tokens.items()
            if all(char in "0123456789.- " for char in token)
        }

        # filter and save str tokens ord() gives the ascii but why we did
        # that if '"' not in why that condition and block in none
        # priintable charcter
        self.str_tokens: dict[int, str] = {
            i: token
            for i, token in self.tokens.items()
            if '"' not in token
            and all(32 <= ord(char) <= 126 for char in token)
        }

        # find and save quote id initialzed to none
        self._quote_id: Optional[int] = None
        for idd, token in self.tokens.items():
            if token == '"':
                self._quote_id = idd
                break

        # filter and save stop tokens that is | , | { | " | newline |.
        # i have a question does that token " ,hello" consider as a stop
        # token and if so is it hello not stop
        self.stop_tokens: dict[int, str] = {
            i: toke
            for i, toke in self.tokens.items()
            if toke.lstrip() and toke.lstrip()[0] in ',}"\n'
        }

    # build the main function that process the prompt
    def process_prompt(
        self, functions: list[functiondef], user_prompt: str
    ) -> Optional[FunctionCallResult]:
        prompt: str = prompt_builder(functions, user_prompt)
        prefix: str = '{"name": "'
        input_ids: list[int] = self.model.encode(prompt + prefix)[0].tolist()

        # using the input ids i should select the function name
        # create a function that do that job (picking the right function name)
        # its a function that pick from a list of things
        fn_name: str = self.generate_function_name(
            input_ids, [f.name for f in functions] + ["fn_none"]
        )
        if fn_name == "fn_none":
            print(
                f"Error: No matching function found "
                f'for prompt: "{user_prompt}"'
            )
            return None

        # use next we generator expretion () to check function by function
        # we generate the function object
        fn: functiondef = next(
            (f for f in functions if f.name == fn_name), functions[0]
        )

        # next step find the arguments
        #  {"name": "fn_add_numbers", "parameters": {
        # args{"a" : 12, "name" : "string", "bool" , True}
        args: dict[str, Any] = {}
        prefix += fn.name + '", "parameters": {'
        # explain this part give examples why we need the index
        for i, (p_name, p_def) in enumerate(fn.parameters.items()):
            sep = ", " if i > 0 else ""
            prefix += sep + f'"{p_name}":'
            # encode everything together so the model generate the correct
            # value of a
            input_ids = self.model.encode(prompt + prefix)[0].tolist()

            # we need to generate the value based on the paremeter type
            val_str: str = ""
            if p_def.type == "boolean":
                val_str = self.generate_function_name(
                    input_ids, [" true", " false", "true", "false"]
                )
                args[p_name] = val_str.strip() == "true"
                prefix += " " + val_str.strip()
            elif p_def.type in ("number", "integer"):
                val_str = self._generate_number(input_ids)
                try:
                    num = float(val_str)
                    if p_def.type == "integer":
                        args[p_name] = int(num)
                    else:
                        args[p_name] = num
                except ValueError:
                    args[p_name] = 0
                prefix += " " + val_str.strip()
            else:
                val_str = self._generate_string(
                    input_ids, param_name=p_name
                ).lstrip()
                if p_name == "path" and "\\\\" in val_str:
                    val_str = val_str.replace("\\\\", "\\")
                    
                # Regex Brute-Force Formatter
                if p_name == "regex":

                
                            
                    # NUMBERS TEST: Fix unclosed parentheses
                    # (Fixes "([0-9]+)\\s([" -> "([0-9]+)\\s")
                    while val_str.count("(") > val_str.count(")"):
                        val_str = val_str[:val_str.rfind("(")]
                        
                    # NUMBERS TEST: Drop trailing whitespace matcher
                    # (Fixes "([0-9]+)\\s" -> "([0-9]+)")
                    if val_str.endswith("\\s"):
                        val_str = val_str[:-2]
                        
                    # CAT TEST & NUMBERS TEST: Drop trailing garbage
                    # (Fixes "cat$|" -> "cat" and strips stray backslashes)
                    while val_str.endswith(("|", "$", "\\")):
                        val_str = val_str[:-1]
                        
                
    

                args[p_name] = val_str
                prefix += ' "' + val_str + '"'

        return FunctionCallResult(
            prompt=user_prompt, name=fn.name, parameters=args
        )

    # i will see that later
    # i need to create a function that generate arg number token by token
    # first i need to check if its a valid number
    def _is_valid_num(self, s: str) -> bool:
        try:
            float(s)
            return True
        except ValueError:
            return False

    def _is_valid_prefix(self, ss: str) -> bool:
        # emty string or start only + or -
        if ss in ("", " -", "-"):
            return True
        # if it end with 2. or 2.. we check the charcter before .
        if ss.endswith("."):
            try:
                float(ss[:-1])
                return True
            except Exception:
                return False

        # anything else we can check it with is valid number
        # 2 3 normal or 2++ or 2-- or those characters
        return self._is_valid_num(ss)

    def _generate_number(self, input_ids: list[int]) -> str:
        generated = ""
        for _ in range(16):
            best_score = float("-inf")
            best_id: Optional[int] = None
            try:
                logits = self.model.get_logits_from_input_ids(input_ids)
            except Exception as e:
                print(f"error : {e}")
                break

            for i, t in self.num_tokens.items():
                # skip scientique notations
                # there is some scientifique numbers that are like this 2e4
                # or 2E4
                if "e" in t or "E" in t:
                    continue
                condidate = generated + t
                # here we need to check if the condidate could still lead
                # to a valid number ex 2. not 2.. not valid prefix
                if self._is_valid_prefix(condidate):
                    if logits[i] > best_score:
                        best_score = logits[i]
                        best_id = i

            # if no valid token was found, stop
            if best_id is None:
                break

            # define a stopping condition
            # check if the model suggest to stop by suggestion a stop token
            # its score bigger than the best score
            if generated and self._is_valid_num(generated):
                best_stop_token = max(logits[idd] for idd in self.stop_tokens)
                if best_stop_token > best_score:
                    break

            # lets add best id to input ids
            generated += self.num_tokens[best_id].strip()

            input_ids.append(best_id)

        # just a fall back
        return generated if self._is_valid_num(generated) else "0.0"

    # lets generate the function that generate string
    # we have a problem small llm sometimes stuck in infinty loop when
    # generating a text
    # example hellohellohellohellohello so we need to check is it duplicated
    # or not

    def is_duplicate(self, s: str) -> bool:
        lenf = len(s)

        # catch short runs like "*****" that the pattern-based
        # check below can't see (it needs length >= 8 to even try)
        if lenf >= 3 and len(set(s[-3:])) == 1:
            return True

        # lp is len pattern
        # we started with as its when we start reputation on pupose till
        # the half
        for lp in range(4, lenf // 2 + 1):
            if s[lenf - lp: lenf] == s[lenf - lp * 2: lenf - lp]:
                return True
        return False

    # we then need to strip the diplicated
    # if it was hellohellohellohello -> hello
    def strip_duplicated(self, s: str) -> str:
        lenf = len(s)

        # collapse short single-char runs first (e.g. "*****" -> "*")
        if lenf >= 3 and len(set(s[-3:])) == 1:
            ch = s[-1]
            i = lenf
            while i > 0 and s[i - 1] == ch:
                i -= 1
            return s[:i] + ch

        for pl in range(4, lenf // 2 + 1):
            pattern = s[lenf - pl: lenf]  # hello is the pattern
            if s[lenf - 2 * pl: lenf - pl] == pattern:
                while s.endswith(pattern) and len(s) > pl:
                    s = s[:-pl]
                return s
        return s

    # include param_name so we know is it regex or normal string and its
    # detault is empty
    def _generate_string(
        self, input_ids: list[int], param_name: str = ""
    ) -> str:
        generated = ""
        # explain what is regex and why we have to create a max char for it
        if param_name == "regex":
            max_char = 12
        else:
            max_char = 80

        # add the quote id to the model so he knows that we are inside a str
        if self._quote_id is not None:
            input_ids.append(self._quote_id)

        # the loop that generate each token
        for _ in range(80):
            # lets get the logists
            try:
                logits = self.model.get_logits_from_input_ids(input_ids)
            except Exception as e:
                print(f"logits error {e}")
                return generated

            # check the reputation
            if len(generated) > max_char:
                return self.strip_duplicated(generated)

            # check if the stop token is in the top 2 atleast that means that
            # the model are suggesting to stop
            if self._quote_id is not None and generated:
                stop_threshold = 2
                count_quote_score = sum(
                    1 for score in logits if score > logits[self._quote_id]
                )
                if count_quote_score < stop_threshold:
                    return generated

            quote_score = (
                logits[self._quote_id]
                if self._quote_id is not None
                else -float("inf")  # as a fall back
            )

            # best score by defualt is quote score cause its beg of a str
            best_score = quote_score
            best_id = self._quote_id
            for i, t in self.str_tokens.items():
                if logits[i] > best_score:
                    best_score = logits[i]
                    best_id = i

            # the end of the string is a quote
            # check the quote score
            if best_id == self._quote_id:
                return generated

            generated += self.str_tokens[best_id]
            input_ids.append(best_id)

            # check if the model is stuck in repuation
            if len(generated) >= 3 and self.is_duplicate(generated):
                return self.strip_duplicated(generated)

        # return the result
        return self.strip_duplicated(generated)

    def generate_function_name(
        self, input_ids: list[int], functions_names: list[str]
    ) -> str:
        generated = ""
        # create a loop that pick every token till it find the whole function
        while True:
            still_valid = [
                f for f in functions_names if f.startswith(generated)
            ]
            if len(still_valid) <= 1:
                # just a fallback
                return (
                    still_valid[0] if still_valid else functions_names[0]
                )

            # start scoring
            try:
                logits = self.model.get_logits_from_input_ids(input_ids)
            except Exception as e:
                print(f"logits error, {e}")
                return still_valid[0]  # fall back

            # Setting best id and best score
            best_id: Optional[int] = None
            best_score = float("-inf")  # cause its the lowest possibile score
            for tid, tk in self.tokens.items():
                maybe = generated + tk
                if any(sv.startswith(maybe) for sv in still_valid):
                    if best_score < logits[tid]:
                        best_id, best_score = tid, logits[tid]

            # what if we didnt find no token best_id = none
            if best_id is None:
                return still_valid[0]  # fall back

            # we found the best id now with the best score
            generated += self.tokens[best_id]
            # we should add that best id to input ids
            input_ids.append(best_id)

            # now return the that generated
            if generated in functions_names:
                return generated
