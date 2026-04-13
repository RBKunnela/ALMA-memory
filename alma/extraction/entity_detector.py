"""
ALMA Entity Detector - Auto-detect people and projects from text.

Adapted from MemPalace entity_detector.py (MIT License).

Two-pass approach:
  Pass 1: scan text, extract entity candidates with signal counts
  Pass 2: score and classify each candidate as person, project, or uncertain

Returns ALMA Entity objects (from alma.graph.store) for direct integration
with the ALMA graph memory system.

Usage:
    from alma.extraction.entity_detector import detect_entities

    entities = detect_entities("Alice said hello to Bob. They discussed the ALMA project.")
    for entity in entities:
        print(f"{entity.name} ({entity.entity_type})")
"""

import re
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from alma.graph.store import Entity
from alma.observability.logging import get_logger

logger = get_logger(__name__)


# ==================== SIGNAL PATTERNS ====================

# Person signals -- things people do
PERSON_VERB_PATTERNS = [
    r"\b{name}\s+said\b",
    r"\b{name}\s+asked\b",
    r"\b{name}\s+told\b",
    r"\b{name}\s+replied\b",
    r"\b{name}\s+laughed\b",
    r"\b{name}\s+smiled\b",
    r"\b{name}\s+cried\b",
    r"\b{name}\s+felt\b",
    r"\b{name}\s+thinks?\b",
    r"\b{name}\s+wants?\b",
    r"\b{name}\s+loves?\b",
    r"\b{name}\s+hates?\b",
    r"\b{name}\s+knows?\b",
    r"\b{name}\s+decided\b",
    r"\b{name}\s+pushed\b",
    r"\b{name}\s+wrote\b",
    r"\bhey\s+{name}\b",
    r"\bthanks?\s+{name}\b",
    r"\bhi\s+{name}\b",
    r"\bdear\s+{name}\b",
]

# Person signals -- pronouns resolving nearby
PRONOUN_PATTERNS = [
    r"\bshe\b",
    r"\bher\b",
    r"\bhers\b",
    r"\bhe\b",
    r"\bhim\b",
    r"\bhis\b",
    r"\bthey\b",
    r"\bthem\b",
    r"\btheir\b",
]

# Person signals -- dialogue markers
DIALOGUE_PATTERNS = [
    r"^>\s*{name}[:\s]",  # > Speaker: ...
    r"^{name}:\s",  # Speaker: ...
    r"^\[{name}\]",  # [Speaker]
    r'"{name}\s+said',
]

# Project signals -- things projects have/do
PROJECT_VERB_PATTERNS = [
    r"\bbuilding\s+{name}\b",
    r"\bbuilt\s+{name}\b",
    r"\bship(?:ping|ped)?\s+{name}\b",
    r"\blaunch(?:ing|ed)?\s+{name}\b",
    r"\bdeploy(?:ing|ed)?\s+{name}\b",
    r"\binstall(?:ing|ed)?\s+{name}\b",
    r"\bthe\s+{name}\s+architecture\b",
    r"\bthe\s+{name}\s+pipeline\b",
    r"\bthe\s+{name}\s+system\b",
    r"\bthe\s+{name}\s+repo\b",
    r"\b{name}\s+v\d+\b",  # MemPal v2
    r"\b{name}\.py\b",  # mempalace.py
    r"\b{name}-core\b",  # mempal-core (hyphen only, not underscore)
    r"\b{name}-local\b",
    r"\bimport\s+{name}\b",
    r"\bpip\s+install\s+{name}\b",
]

# Words that are almost certainly NOT entities
STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "by",
    "from",
    "as",
    "is",
    "was",
    "are",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "must",
    "shall",
    "can",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "they",
    "them",
    "their",
    "we",
    "our",
    "you",
    "your",
    "i",
    "my",
    "me",
    "he",
    "she",
    "his",
    "her",
    "who",
    "what",
    "when",
    "where",
    "why",
    "how",
    "which",
    "if",
    "then",
    "so",
    "not",
    "no",
    "yes",
    "ok",
    "okay",
    "just",
    "very",
    "really",
    "also",
    "already",
    "still",
    "even",
    "only",
    "here",
    "there",
    "now",
    "too",
    "up",
    "out",
    "about",
    "like",
    "use",
    "get",
    "got",
    "make",
    "made",
    "take",
    "put",
    "come",
    "go",
    "see",
    "know",
    "think",
    "true",
    "false",
    "none",
    "null",
    "new",
    "old",
    "all",
    "any",
    "some",
    "return",
    "print",
    "def",
    "class",
    "import",
    # Common capitalized words in prose that aren't entities
    "step",
    "usage",
    "run",
    "check",
    "find",
    "add",
    "set",
    "list",
    "args",
    "dict",
    "str",
    "int",
    "bool",
    "path",
    "file",
    "type",
    "name",
    "note",
    "example",
    "option",
    "result",
    "error",
    "warning",
    "info",
    "every",
    "each",
    "more",
    "less",
    "next",
    "last",
    "first",
    "second",
    "stack",
    "layer",
    "mode",
    "test",
    "stop",
    "start",
    "copy",
    "move",
    "source",
    "target",
    "output",
    "input",
    "data",
    "item",
    "key",
    "value",
    "returns",
    "raises",
    "yields",
    "self",
    "cls",
    "kwargs",
    # Common sentence-starting / abstract words that aren't entities
    "world",
    "well",
    "want",
    "topic",
    "choose",
    "social",
    "cars",
    "phones",
    "healthcare",
    "ex",
    "machina",
    "deus",
    "human",
    "humans",
    "people",
    "things",
    "something",
    "nothing",
    "everything",
    "anything",
    "someone",
    "everyone",
    "anyone",
    "way",
    "time",
    "day",
    "life",
    "place",
    "thing",
    "part",
    "kind",
    "sort",
    "case",
    "point",
    "idea",
    "fact",
    "sense",
    "question",
    "answer",
    "reason",
    "number",
    "version",
    "system",
    # Greetings and filler words at sentence starts
    "hey",
    "hi",
    "hello",
    "thanks",
    "thank",
    "right",
    "let",
    # UI/action words that appear in how-to content
    "click",
    "hit",
    "press",
    "tap",
    "drag",
    "drop",
    "open",
    "close",
    "save",
    "load",
    "launch",
    "install",
    "download",
    "upload",
    "scroll",
    "select",
    "enter",
    "submit",
    "cancel",
    "confirm",
    "delete",
    "paste",
    "write",
    "read",
    "search",
    "show",
    "hide",
    # Common filesystem/technical capitalized words
    "desktop",
    "documents",
    "downloads",
    "users",
    "home",
    "library",
    "applications",
    "preferences",
    "settings",
    "terminal",
    # Abstract/topic words
    "actor",
    "vector",
    "remote",
    "control",
    "duration",
    "fetch",
    # Abstract concepts that appear as subjects but aren't entities
    "agents",
    "tools",
    "others",
    "guards",
    "ethics",
    "regulation",
    "learning",
    "thinking",
    "memory",
    "language",
    "intelligence",
    "technology",
    "society",
    "culture",
    "future",
    "history",
    "science",
    "model",
    "models",
    "network",
    "networks",
    "training",
    "inference",
}


# ==================== CANDIDATE EXTRACTION ====================


def _extract_candidates(text: str) -> Dict[str, int]:
    """
    Extract all capitalized proper noun candidates from text.

    Finds single capitalized words (e.g. "Alice") and multi-word
    proper nouns (e.g. "Memory Palace") that appear 3+ times.

    Args:
        text: The text to extract candidates from.

    Returns:
        Dict mapping candidate name to frequency count.
    """
    # Find all capitalized words
    raw = re.findall(r"\b([A-Z][a-z]{1,19})\b", text)

    counts: Dict[str, int] = defaultdict(int)
    for word in raw:
        if word.lower() not in STOPWORDS and len(word) > 1:
            counts[word] += 1

    # Also find multi-word proper nouns (e.g. "Memory Palace", "Claude Code")
    multi = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", text)
    for phrase in multi:
        if not any(w.lower() in STOPWORDS for w in phrase.split()):
            counts[phrase] += 1

    # Filter: must appear at least 3 times to be a candidate
    return {name: count for name, count in counts.items() if count >= 3}


# ==================== SIGNAL SCORING ====================


def _build_patterns(name: str) -> Dict:
    """
    Pre-compile all regex patterns for a single entity name.

    Args:
        name: The entity name to build patterns for.

    Returns:
        Dict of compiled pattern groups keyed by signal type.
    """
    n = re.escape(name)
    return {
        "dialogue": [
            re.compile(p.format(name=n), re.MULTILINE | re.IGNORECASE)
            for p in DIALOGUE_PATTERNS
        ],
        "person_verbs": [
            re.compile(p.format(name=n), re.IGNORECASE) for p in PERSON_VERB_PATTERNS
        ],
        "project_verbs": [
            re.compile(p.format(name=n), re.IGNORECASE) for p in PROJECT_VERB_PATTERNS
        ],
        "direct": re.compile(
            rf"\bhey\s+{n}\b|\bthanks?\s+{n}\b|\bhi\s+{n}\b", re.IGNORECASE
        ),
        "versioned": re.compile(rf"\b{n}[-v]\w+", re.IGNORECASE),
        "code_ref": re.compile(rf"\b{n}\.(py|js|ts|yaml|yml|json|sh)\b", re.IGNORECASE),
    }


def _score_entity(name: str, text: str, lines: List[str]) -> Dict:
    """
    Score a candidate entity as person vs project.

    Examines dialogue markers, person verbs, pronoun proximity, direct address,
    project verbs, versioned references, and code file references to produce
    a person_score and project_score.

    Args:
        name: The candidate entity name.
        text: The full combined text.
        lines: The text split into lines.

    Returns:
        Dict with person_score, project_score, person_signals, project_signals.
    """
    patterns = _build_patterns(name)
    person_score = 0
    project_score = 0
    person_signals: List[str] = []
    project_signals: List[str] = []

    # --- Person signals ---

    # Dialogue markers (strong signal)
    for rx in patterns["dialogue"]:
        matches = len(rx.findall(text))
        if matches > 0:
            person_score += matches * 3
            person_signals.append(f"dialogue marker ({matches}x)")

    # Person verbs
    for rx in patterns["person_verbs"]:
        matches = len(rx.findall(text))
        if matches > 0:
            person_score += matches * 2
            person_signals.append(f"'{name} ...' action ({matches}x)")

    # Pronoun proximity -- pronouns within 3 lines of the name
    name_lower = name.lower()
    name_line_indices = [
        i for i, line in enumerate(lines) if name_lower in line.lower()
    ]
    pronoun_hits = 0
    for idx in name_line_indices:
        window_text = " ".join(lines[max(0, idx - 2) : idx + 3]).lower()
        for pronoun_pattern in PRONOUN_PATTERNS:
            if re.search(pronoun_pattern, window_text):
                pronoun_hits += 1
                break
    if pronoun_hits > 0:
        person_score += pronoun_hits * 2
        person_signals.append(f"pronoun nearby ({pronoun_hits}x)")

    # Direct address
    direct = len(patterns["direct"].findall(text))
    if direct > 0:
        person_score += direct * 4
        person_signals.append(f"addressed directly ({direct}x)")

    # --- Project signals ---

    for rx in patterns["project_verbs"]:
        matches = len(rx.findall(text))
        if matches > 0:
            project_score += matches * 2
            project_signals.append(f"project verb ({matches}x)")

    versioned = len(patterns["versioned"].findall(text))
    if versioned > 0:
        project_score += versioned * 3
        project_signals.append(f"versioned/hyphenated ({versioned}x)")

    code_ref = len(patterns["code_ref"].findall(text))
    if code_ref > 0:
        project_score += code_ref * 3
        project_signals.append(f"code file reference ({code_ref}x)")

    return {
        "person_score": person_score,
        "project_score": project_score,
        "person_signals": person_signals,
        "project_signals": project_signals,
    }


# ==================== CLASSIFY ====================


def _classify_entity(name: str, frequency: int, scores: Dict) -> Dict:
    """
    Given scores, classify as person / project / uncertain.

    Uses a two-signal-category requirement for confident person classification
    to avoid false positives from single signal types with many hits.

    Args:
        name: The candidate entity name.
        frequency: How many times the name appeared.
        scores: The score dict from _score_entity.

    Returns:
        Dict with name, type, confidence, frequency, and signals.
    """
    ps = scores["person_score"]
    prs = scores["project_score"]
    total = ps + prs

    if total == 0:
        # No strong signals -- frequency-only candidate, uncertain
        confidence = min(0.4, frequency / 50)
        return {
            "name": name,
            "type": "uncertain",
            "confidence": round(confidence, 2),
            "frequency": frequency,
            "signals": [f"appears {frequency}x, no strong type signals"],
        }

    person_ratio = ps / total if total > 0 else 0

    # Require TWO different signal categories to confidently classify as a person.
    signal_categories = set()
    for s in scores["person_signals"]:
        if "dialogue" in s:
            signal_categories.add("dialogue")
        elif "action" in s:
            signal_categories.add("action")
        elif "pronoun" in s:
            signal_categories.add("pronoun")
        elif "addressed" in s:
            signal_categories.add("addressed")

    has_two_signal_types = len(signal_categories) >= 2

    if person_ratio >= 0.7 and has_two_signal_types and ps >= 5:
        entity_type = "person"
        confidence = min(0.99, 0.5 + person_ratio * 0.5)
        signals = scores["person_signals"][:3] or [f"appears {frequency}x"]
    elif person_ratio >= 0.7 and (not has_two_signal_types or ps < 5):
        # Pronoun-only match -- downgrade to uncertain
        entity_type = "uncertain"
        confidence = 0.4
        signals = scores["person_signals"][:3] + [
            f"appears {frequency}x — pronoun-only match"
        ]
    elif person_ratio <= 0.3:
        entity_type = "project"
        confidence = min(0.99, 0.5 + (1 - person_ratio) * 0.5)
        signals = scores["project_signals"][:3] or [f"appears {frequency}x"]
    else:
        entity_type = "uncertain"
        confidence = 0.5
        signals = (scores["person_signals"] + scores["project_signals"])[:3]
        signals.append("mixed signals — needs review")

    return {
        "name": name,
        "type": entity_type,
        "confidence": round(confidence, 2),
        "frequency": frequency,
        "signals": signals,
    }


# ==================== PUBLIC API ====================


def detect_entities(text: str, min_frequency: int = 3) -> List[Entity]:
    """
    Detect people and projects from text using regex heuristics.

    Uses a two-pass approach: first extracts capitalized proper noun candidates
    that appear frequently, then scores each candidate against person/project
    signal patterns to classify them.

    Args:
        text: The text to scan for entities.
        min_frequency: Minimum number of appearances to be a candidate.
            Defaults to 3.

    Returns:
        List of ALMA Entity objects with entity_type set to
        "person", "project", or "uncertain".

    Example:
        >>> entities = detect_entities("Alice told Bob about the ALMA project.")
        >>> for e in entities:
        ...     print(f"{e.name} ({e.entity_type})")
    """
    if not text or not text.strip():
        return []

    lines = text.splitlines()
    candidates = _extract_candidates(text)

    if not candidates:
        logger.info("No entity candidates found in text")
        return []

    entities: List[Entity] = []

    for name, frequency in sorted(candidates.items(), key=lambda x: x[1], reverse=True):
        if frequency < min_frequency:
            continue

        scores = _score_entity(name, text, lines)
        classified = _classify_entity(name, frequency, scores)

        entity = Entity(
            id=f"detected-{uuid.uuid4().hex[:12]}",
            name=classified["name"],
            entity_type=classified["type"],
            properties={
                "confidence": classified["confidence"],
                "frequency": classified["frequency"],
                "signals": classified["signals"],
                "detection_method": "regex_heuristic",
            },
            created_at=datetime.now(timezone.utc),
        )
        entities.append(entity)

    # Sort by confidence descending
    entities.sort(key=lambda e: e.properties.get("confidence", 0), reverse=True)

    logger.info(
        f"Detected {len(entities)} entities from text "
        f"({sum(1 for e in entities if e.entity_type == 'person')} people, "
        f"{sum(1 for e in entities if e.entity_type == 'project')} projects, "
        f"{sum(1 for e in entities if e.entity_type == 'uncertain')} uncertain)"
    )

    return entities


def detect_entities_from_file(filepath: str, max_bytes: int = 50_000) -> List[Entity]:
    """
    Detect entities from a file.

    Reads the file content (up to max_bytes) and runs entity detection
    on the text.

    Args:
        filepath: Path to the file to scan.
        max_bytes: Maximum bytes to read from the file. Defaults to 50KB.

    Returns:
        List of ALMA Entity objects detected in the file.

    Raises:
        FileNotFoundError: If the file does not exist.
        OSError: If the file cannot be read.

    Example:
        >>> entities = detect_entities_from_file("notes/meeting.md")
        >>> print(f"Found {len(entities)} entities")
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    content = path.read_text(encoding="utf-8", errors="replace")[:max_bytes]
    return detect_entities(content)
