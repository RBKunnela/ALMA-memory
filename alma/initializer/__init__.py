"""
ALMA Initializer Module.

Bootstrap pattern that orients the agent before work begins.

Usage:
    from alma.initializer import SessionInitializer, InitializationResult

    initializer = SessionInitializer(alma)
    result = initializer.initialize(
        project_id="my-project",
        agent="Helena",
        user_prompt="Test the login flow",
        project_path="/path/to/project",
    )

    # Inject into agent prompt
    prompt = f'''
    {result.to_prompt()}

    Now proceed with the first work item.
    '''
"""

from alma.initializer.initializer import SessionInitializer
from alma.initializer.types import (
    CodebaseOrientation,
    InitializationResult,
    RulesOfEngagement,
)

__all__ = [
    "CodebaseOrientation",
    "InitializationResult",
    "RulesOfEngagement",
    "SessionInitializer",
]
