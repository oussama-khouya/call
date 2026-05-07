*This project has been created as part of the 42 curriculum by okhouya.*

# Call Me Maybe - Function Calling with Constrained Decoding

## Description

This project implements a **function calling system** that translates natural language prompts into structured, machine-executable function calls using a small language model (Qwen3-0.6B, ~500M parameters).

The key innovation is **constrained decoding**: instead of hoping the model produces valid JSON, we actively guide token generation at each step to *guarantee* 100% valid, schema-compliant output. This technique achieves near-perfect reliability even with a tiny model.

### How It Works

Given a prompt like *"What is the sum of 40 and 2?"*, the system outputs:
```json
{
  "prompt": "What is the sum of 40 and 2?",
  "name": "fn_add_numbers",
  "parameters": {"a": 40.0, "b": 2.0}
}
```

## Instructions

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

```bash
make install
# or manually:
uv sync
```

### Running

```bash
# Default (uses files in data/input/)
make run

# With custom paths
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

### Linting

```bash
make lint          # flake8 + mypy (required flags)
make lint-strict   # flake8 + mypy --strict
```

### Cleaning

```bash
make clean
```

## Algorithm Explanation

### Constrained Decoding

Standard LLM generation picks the highest-probability next token at each step. Constrained decoding modifies this by:

1. **Get logits**: The model produces a probability distribution over all ~150k tokens in its vocabulary
2. **Compute valid tokens**: Based on the current JSON state and schema, determine which tokens are valid
3. **Mask invalid tokens**: Set logits of invalid tokens to `-infinity`
4. **Select**: Pick the highest-probability token from the remaining valid set
5. **Repeat**: Add the selected token and continue

This guarantees that every generated token maintains valid JSON structure and schema compliance.

### Function Selection

The function name is selected using a constrained choice over the exact set of available function names. At each token step, only tokens that continue a valid function name prefix are allowed.

### Argument Generation

For each parameter:
- **Numbers**: Only digits, decimal points, and minus signs are allowed
- **Strings**: An opening quote is forced, content is generated freely, and a closing quote terminates the value
- **Booleans**: Constrained to exactly `true` or `false`

## Design Decisions

1. **Two-phase generation**: Function name selection and argument generation are separated for better reliability
2. **Vocabulary-based masking**: We load the full tokenizer vocabulary and check each token's decoded string against JSON/schema constraints
3. **Pydantic validation**: All data models use pydantic for runtime validation as required
4. **Greedy decoding**: We always pick the highest-probability valid token (no sampling/temperature) for maximum determinism

## Performance Analysis

- **Accuracy**: 90%+ correct function selection and argument extraction on standard prompts
- **JSON validity**: 100% - constrained decoding guarantees valid, parseable JSON
- **Speed**: All 11 test prompts process in under 5 minutes on standard hardware
- **Reliability**: Graceful error handling for malformed inputs, missing files, and edge cases

## Challenges Faced

1. **Tokenizer vocabulary format**: Different tokenizers store vocab differently. We handle the Qwen tokenizer's `tokenizer.json` format with its BPE vocabulary
2. **Token normalization**: The BPE tokenizer uses special characters (Ġ for space, Ċ for newline) that must be normalized before string matching
3. **Multi-token values**: Numbers and strings may span multiple tokens, requiring careful state tracking
4. **Performance**: Iterating over 150k+ tokens at each step requires efficient numpy operations

## Testing Strategy

- Validate JSON output is always parseable
- Check function names match available definitions
- Verify argument types match the schema (numbers are floats, strings are strings)
- Test with edge cases: empty strings, large numbers, special characters
- Test with different function definition sets to ensure no hardcoding

## Example Usage

```bash
# Run with default input files
uv run python -m src

# Example output (data/output/function_calling_results.json):
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
  },
  {
    "prompt": "Greet shrek",
    "name": "fn_greet",
    "parameters": {"name": "shrek"}
  }
]
```

## Resources

- [Constrained Decoding for LLMs](https://huggingface.co/blog/constrained-beam-search) - Hugging Face blog on constrained generation
- [Outlines library](https://github.com/dottxt-ai/outlines) - Reference implementation of structured generation (not used, for learning only)
- [JSON Schema](https://json-schema.org/) - Understanding structured output formats
- [Qwen3-0.6B Model](https://huggingface.co/Qwen/Qwen3-0.6B) - The model used in this project

### AI Usage

AI was used to:
- Research constrained decoding techniques and best practices
- Help debug tokenizer vocabulary parsing edge cases
- Generate initial test cases for validation
- Review code structure and suggest improvements
