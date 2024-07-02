import re


def find_variable_assignments(text):
    """
    This function uses regular expressions to find patterns where a variable (which can include leading/trailing whitespace)
    is followed by an equals sign ('='), and then another variable name or value until a newline or comma.
    """
    # The regex pattern looks for variable names (which can include underscores and alphanumeric characters)
    # Allows for any amount of whitespace before and after the variable names and '='
    # Stops matching at a newline or comma
    pattern = r'\s*(\w+)\s*=\s*(\w+)[,\n]'

    # Find all matches in the text
    matches = re.findall(pattern, text)

    return matches


# Example usage:
input_text = "   var1 = value1,\nvar2 = value2,   var3=   value3,\nend"
assignments_found = find_variable_assignments(input_text)
print(assignments_found)
input_text2 = "   var1 ~ value1,"
assignments_found = find_variable_assignments(input_text2)
print(assignments_found)


def replace_variable_assignments(text, replacement="REPLACED"):
    """
    This function uses regular expressions to find patterns of variable assignments
    and replaces each match with a predefined string.
    """
    # The regex pattern remains the same as before
    pattern = r'\s*(\w+)\s*=\s*(\w+)[,\n]'

    # Use re.sub to substitute each match with the replacement string
    replaced_text = re.sub(pattern, replacement, text)

    return replaced_text


# Example usage:
input_text = "   var1 = value1,\nvar2 = value2,\n   var3=   value3,\nend"
modified_text = replace_variable_assignments(input_text)
print(modified_text)


def replace_variable_assignments_keep_whitespace(text, replacement="REPLACED"):
    """
    This function uses regular expressions to find patterns of variable assignments,
    preserving the leading whitespace, and replaces the rest of the matched pattern
    with a predefined string.
    """
    # The regex pattern now includes a capturing group for leading whitespace
    pattern = r'(\s*)(\w+)\s*=\s*(\w+)[,\n]'

    # In the replacement string, '\1' refers to the first captured group (the leading whitespace)
    # The rest of the matched pattern is replaced with the predefined string
    replaced_text = re.sub(pattern, r'\1' + replacement, text)

    return replaced_text


# Example usage:
input_text = "   var1 = value1,\nvar2 = value2,\n   var3=   value3,\nend"
modified_text = replace_variable_assignments_keep_whitespace(input_text)
print('3: ')
print(modified_text)
