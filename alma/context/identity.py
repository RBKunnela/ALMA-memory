"""
Layer 0: Identity File Management for ALMA MemoryStack.

Inspired by MemPalace layers.py (MIT License).

Manages a plain-text identity file that provides the agent's core
identity context (~100 tokens). Always loaded at wake-up.

Example identity.txt:
    I am Atlas, a personal AI assistant for Alice.
    Traits: warm, direct, remembers everything.
    People: Alice (creator), Bob (Alice's partner).
    Project: A journaling app that helps people process emotions.
"""

import os
from pathlib import Path
from typing import Optional

from alma.observability.logging import get_logger
from alma.utils.tokenizer import estimate_tokens_simple

logger = get_logger(__name__)

# Default location for the identity file
DEFAULT_IDENTITY_PATH = os.path.join(os.path.expanduser("~"), ".alma", "identity.txt")

# Default identity content when no file exists
DEFAULT_IDENTITY_TEXT = (
    "No identity configured. "
    "Create ~/.alma/identity.txt with your agent's identity description."
)


class IdentityManager:
    """
    Manages Layer 0 identity for the MemoryStack.

    Reads a plain-text identity file and caches its contents.
    Provides token estimation for budget tracking.

    Args:
        identity_path: Path to the identity file.
            Defaults to ~/.alma/identity.txt.
        create_default: If True, creates the identity file with
            default content when it does not exist.
    """

    def __init__(
        self,
        identity_path: Optional[str] = None,
        create_default: bool = False,
    ):
        self.path = identity_path or DEFAULT_IDENTITY_PATH
        self._text: Optional[str] = None
        self._create_default = create_default

    def load(self) -> str:
        """
        Load and return the identity text.

        Reads from disk on first call, then caches. Returns a
        sensible default if the file does not exist.

        Returns:
            The identity text string.
        """
        if self._text is not None:
            return self._text

        path = Path(self.path)
        if path.exists():
            try:
                self._text = path.read_text(encoding="utf-8").strip()
                logger.info(
                    "Identity loaded",
                    path=self.path,
                    tokens=self.token_count,
                )
            except OSError as exc:
                logger.warning(
                    "Failed to read identity file, using default",
                    path=self.path,
                    error=str(exc),
                )
                self._text = DEFAULT_IDENTITY_TEXT
        else:
            if self._create_default:
                self._write_default(path)
                self._text = DEFAULT_IDENTITY_TEXT
            else:
                self._text = DEFAULT_IDENTITY_TEXT

        return self._text

    @property
    def text(self) -> str:
        """Return the identity text, loading if necessary."""
        return self.load()

    @property
    def token_count(self) -> int:
        """Estimate the token count of the identity text."""
        return estimate_tokens_simple(self.load())

    def reload(self) -> str:
        """
        Force reload identity from disk (clears cache).

        Returns:
            The freshly loaded identity text.
        """
        self._text = None
        return self.load()

    def _write_default(self, path: Path) -> None:
        """Create the identity file with default content."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(DEFAULT_IDENTITY_TEXT, encoding="utf-8")
            logger.info(
                "Created default identity file",
                path=str(path),
            )
        except OSError as exc:
            logger.warning(
                "Failed to create default identity file",
                path=str(path),
                error=str(exc),
            )
