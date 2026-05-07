# ABOUTME: Builds the prompt string sent to the LLM for function selection.
# ABOUTME: Formats available functions and user query into a prompt.

from src.models import FunctionDefinition


def build_function_selection_prompt(
    functions: list[FunctionDefinition],
    user_prompt: str,
) -> str:
    """Build a prompt that asks the LLM to select which function to call.

    The prompt lists all available functions with their descriptions and
    parameter schemas, then presents the user's query. The LLM is expected
    to respond with a JSON object containing the function name and arguments.

    Args:
        functions: List of available function definitions.
        user_prompt: The natural language request from the user.

    Returns:
        A formatted prompt string for the LLM.
    """
    func_descriptions: list[str] = []
    for fn in functions:
        params_str = ", ".join(
            f"{pname}: {pdef.type}" for pname, pdef in fn.parameters.items()
        )
        func_descriptions.append(
            f"- {fn.name}({params_str}): {fn.description}"
        )

    functions_block = "\n".join(func_descriptions)

    prompt = (
        f"You are a function calling assistant. "
        f"Given the user request, select the correct function and extract "
        f"its arguments.\n\n"
        f"Available functions:\n{functions_block}\n\n"
        f"User request: {user_prompt}\n\n"
        f"Respond with a JSON object containing \"name\" and \"parameters\".\n"
        f"Example: {{\"name\": \"fn_add\", \"parameters\": {{\"a\": 2}}}}\n\n"
        f"JSON response:\n"
    )
    return prompt
