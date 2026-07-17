from src.models import functiondef

def prompt_builder(functions : list[functiondef], user_prompt) -> str:

    # we need first to build a function block that have 
    # the function name (the parameters with its types ) and function decription 
    # [ add ("a": int , "b" : int)  Adds two numbers , and other function and so one] list of strs

    function_lines = []
    # we need function by function 
    for fn in functions:
        # we need to get parameters first 
        param = " ,".join(f"{name} : {p_def.type}" for name , p_def in fn.parameters.items())
        function_lines.append(f"- {fn.name} ({param}) : {fn.description}")

    #in the end i want them to become blocks sep by new line
    functions_block = "\n".join(function_lines)
    return (
        f"You are a function calling assistant. "
        f"Given the user request, select the correct function and extract "
        f"its arguments.\n\n"
        f"Available functions:\n{functions_block}\n\n"
        f"Rules for regex parameters:\n"
        f"- Use simple, short regex patterns\n"
        f"- To match all numbers: [0-9]+\n"
        f"- To match all vowels: [aeiouAEIOU]\n"
        f"- To match a word exactly: cat\n"
        f"- replacement is the exact literal replacement string\n\n"
        f"- If the user request does not match any available function, "
        f"use fn_none with no parameters\n\n"
        f"User request: {user_prompt}\n\n"
        f"Respond with only a JSON object.\n"
        f"Examples:\n"
        f"{{\"name\": \"fn_add_numbers\", "
        f"\"parameters\": {{\"a\": 5, \"b\": 3}}}}\n"
        f"{{\"name\": \"fn_substitute_string_with_regex\", "
        f"\"parameters\": {{\"source_string\": \"abc 123\", "
        f"\"regex\": \"[0-9]+\", \"replacement\": \"NUMBERS\"}}}}\n"
        f"{{\"name\": \"fn_substitute_string_with_regex\", "
        f"\"parameters\": {{\"source_string\": \"hello\", "
        f"\"regex\": \"[aeiouAEIOU]\", \"replacement\": \"*\"}}}}\n\n"
        f"JSON response:\n"
    )

        
