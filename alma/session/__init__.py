"""
ALMA Session Management Module.

Handles session continuity, handoffs, and quick context reload.
"""

from alma.session.manager import SessionManager
from alma.session.types import (
    SessionContext,
    SessionHandoff,
    SessionOutcome,
)

__all__ = [
    "SessionHandoff",
    "SessionContext",
    "SessionOutcome",
    "SessionManager",
]
