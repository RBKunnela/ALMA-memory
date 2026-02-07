"""
ALMA Prompt Sanitization Utilities.

Prevents format string injection in LLM prompt templates.
"""


def sanitize_for_prompt(text: str) -> str:
    """Escape curly braces in text before inserting into .format() templates.

    This prevents user-controlled data from being interpreted as
    format specifiers (e.g., ``{0}``, ``{key}``) which could cause
    KeyError/IndexError or unintended substitution.

    Args:
        text: Raw text that may contain curly braces.

    Returns:
        Text with ``{`` and ``}`` doubled so .format() treats them as literals.
    """
    return text.replace("{", "{{").replace("}", "}}")
