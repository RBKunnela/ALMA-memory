"""
ALMA Session Management Module.

Handles session continuity, handoffs, and quick context reload.
"""

from alma.session.types import (
    SessionHandoff,
    SessionContext,
    SessionOutcome,
)
from alma.session.manager import SessionManager

__all__ = [
    "SessionHandoff",
    "SessionContext",
    "SessionOutcome",
    "SessionManager",
]
