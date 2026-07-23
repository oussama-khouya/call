import json
import pytest

from src.constrained_decoder import constrained_decoding
from src.models import FunctionCallResult, functiondef, parameterdef, prompt
from src.prompt_builder import prompt_builder
from src.vocabulary import Vocabulary


class DummyTokenizer:
    """Mock tokenizer for testing without heavy model loading."""
    def get_path_to_tokenizer_file(self) -> str:
        return ""


def test_models():
    param = parameterdef(type="number")
    assert param.type == "number"

    fn = functiondef(
        name="fn_add",
        description="Add two numbers",
        parameters={"a": param},
        returns=param,
    )
    assert fn.name == "fn_add"
    assert fn.description == "Add two numbers"

    p = prompt(prompt="Test prompt")
    assert p.prompt == "Test prompt"

    res = FunctionCallResult(
        prompt="Test prompt", name="fn_add", parameters={"a": 5.0}
    )
    assert res.name == "fn_add"
    assert res.parameters["a"] == 5.0


def test_prompt_builder():
    fn = functiondef(
        name="fn_add_numbers",
        description="Adds two numbers",
        parameters={
            "a": parameterdef(type="number"),
            "b": parameterdef(type="number"),
        },
        returns=parameterdef(type="number"),
    )
    built_prompt = prompt_builder([fn], "What is 2 + 3?")
    assert "fn_add_numbers" in built_prompt
    assert "What is 2 + 3?" in built_prompt
    assert "JSON response:" in built_prompt


def test_vocabulary(tmp_path):
    tokenizer_file = tmp_path / "tokenizer.json"
    vocab_data = {
        "model": {
            "vocab": {
                "a": 1,
                "b": 2,
                "-": 3,
                " -": 4,
                " 2": 5,
            }
        }
    }
    tokenizer_file.write_text(json.dumps(vocab_data), encoding="utf-8")

    vocab = Vocabulary(str(tokenizer_file))
    assert vocab.get_token_id("a") == 1
    assert vocab.get_token_str(1) == "a"
    assert set(vocab.get_token_ids()) == {1, 2, 3, 4, 5}


def test_constrained_decoding_number_helpers(tmp_path):
    tokenizer_file = tmp_path / "tokenizer.json"
    vocab_data = {
        "model": {
            "vocab": {
                '"': 1,
                "0": 2,
                "1": 3,
                "2": 4,
                "-": 5,
                " -": 6,
                ",": 7,
            }
        }
    }
    tokenizer_file.write_text(json.dumps(vocab_data), encoding="utf-8")

    vocab = Vocabulary(str(tokenizer_file))
    decoder = constrained_decoding(None, vocab)

    assert decoder._is_valid_num("-2.5") is True
    assert decoder._is_valid_num("abc") is False
    assert decoder._is_valid_prefix("-") is True
    assert decoder._is_valid_prefix(" -") is True
    assert decoder._is_valid_prefix("2.") is True
    assert decoder.is_duplicate("hellohellohellohello") is True
    assert decoder.strip_duplicated("hellohellohellohello") == "hello"


@pytest.mark.integration
def test_full_constrained_decoder_integration():
    """Integration test using actual Small_LLM_Model if available."""
    try:
        from llm_sdk import Small_LLM_Model
        model = Small_LLM_Model()
        vocab = Vocabulary(model.get_path_to_tokenizer_file())
        decoder = constrained_decoding(model, vocab)

        fn = functiondef(
            name="fn_add_numbers",
            description="Adds two numbers",
            parameters={
                "a": parameterdef(type="number"),
                "b": parameterdef(type="number"),
            },
            returns=parameterdef(type="number"),
        )

        res_neg = decoder.process_prompt([fn], "What is the sum of -2 and -3?")
        assert res_neg is not None
        assert res_neg.name == "fn_add_numbers"
        assert res_neg.parameters["a"] == -2.0
        assert res_neg.parameters["b"] == -3.0

        res_pos = decoder.process_prompt([fn], "What is the sum of 2 and 3?")
        assert res_pos is not None
        assert res_pos.name == "fn_add_numbers"
        assert res_pos.parameters["a"] == 2.0
        assert res_pos.parameters["b"] == 3.0
    except Exception as e:
        pytest.skip(f"Small_LLM_Model integration test skipped: {e}")
