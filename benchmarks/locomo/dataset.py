"""
LoCoMo Dataset Loader

Loads the LoCoMo (Long Conversational Memory) benchmark dataset from a local
JSON file or downloads `locomo10.json` from the snap-research GitHub repo.

Dataset structure (per conversation):
    - sample_id / conv_id: str -- unique conversation identifier
    - conversation:
        - session_{N}: list of {"speaker": str, "dia_id": str, "text": str, ...}
        - session_{N}_date_time: str -- date context for the session
    - qa: list of
        - question: str
        - answer: str
        - category: int (1=single-hop, 2=multi-hop, 3=temporal,
                         4=open-domain, 5=adversarial)
        - evidence: list[str] -- turn IDs like "D1:3", "D5:12"
        - adversarial_answer: str (for category 5)

Evidence turn-id format: ``"D{session}:{turn}"`` e.g. ``"D2:7"`` means
session 2, turn index 7.

Upstream JSON (v1.0): ``https://github.com/snap-research/locomo/raw/main/data/locomo10.json``
"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

LOCOMO_JSON_URL = "https://github.com/snap-research/locomo/raw/main/data/locomo10.json"

DEFAULT_CACHE_DIR = (
    Path(os.environ.get("ALMA_BENCHMARK_DATA", "/tmp/alma-benchmark-data")) / "locomo"
)
DEFAULT_CACHE_FILE = "locomo10.json"

# Category ID -> human-readable label (per LoCoMo paper)
CATEGORY_LABELS: Dict[int, str] = {
    1: "single_hop",
    2: "multi_hop",
    3: "temporal",
    4: "open_domain",
    5: "adversarial",
}

# Reverse lookup
CATEGORY_IDS: Dict[str, int] = {v: k for k, v in CATEGORY_LABELS.items()}

ALL_CATEGORIES: List[str] = list(CATEGORY_LABELS.values())


@dataclass
class LoCoMoTurn:
    """A single turn within a LoCoMo session."""

    turn_id: str  # e.g. "D1:3" -- session:turn index
    session: int  # 1-based session number
    turn_index: int  # 0-based within session
    speaker: str
    text: str
    date_time: str = ""  # session-level date (repeated on each turn)


@dataclass
class LoCoMoQA:
    """A single LoCoMo question-answer pair."""

    qa_id: str  # synthetic -- {conv_id}_qa_{i}
    question: str
    answer: str
    category: str  # single_hop | multi_hop | temporal | open_domain | adversarial
    category_id: int
    evidence_turn_ids: List[str] = field(default_factory=list)
    adversarial_answer: str = ""

    @property
    def is_adversarial(self) -> bool:
        return self.category == "adversarial"


@dataclass
class LoCoMoConversation:
    """A full LoCoMo multi-session conversation with its QA pairs."""

    conv_id: str
    sessions: Dict[int, List[LoCoMoTurn]] = field(default_factory=dict)
    session_dates: Dict[int, str] = field(default_factory=dict)
    qa_pairs: List[LoCoMoQA] = field(default_factory=list)

    def iter_turns(self):
        """Yield every turn across all sessions in order."""
        for session_num in sorted(self.sessions.keys()):
            for turn in self.sessions[session_num]:
                yield turn

    def turn_by_id(self, turn_id: str) -> Optional[LoCoMoTurn]:
        """Lookup a turn by its ``"D{session}:{turn}"`` evidence ID."""
        for turn in self.iter_turns():
            if turn.turn_id == turn_id:
                return turn
        return None

    @property
    def num_turns(self) -> int:
        return sum(len(ts) for ts in self.sessions.values())

    @property
    def num_sessions(self) -> int:
        return len(self.sessions)


@dataclass
class LoCoMoDataset:
    """
    The complete LoCoMo benchmark.

    Default v1 release contains 10 conversations with ~300 QA pairs total.
    """

    conversations: List[LoCoMoConversation] = field(default_factory=list)
    source_path: str = ""

    def __len__(self) -> int:
        return len(self.conversations)

    def __getitem__(self, idx: int) -> LoCoMoConversation:
        return self.conversations[idx]

    @property
    def total_qa_pairs(self) -> int:
        return sum(len(c.qa_pairs) for c in self.conversations)

    def summary(self) -> str:
        lines = [
            f"LoCoMo Dataset: {len(self)} conversations, {self.total_qa_pairs} QA pairs"
        ]
        cat_counts: Dict[str, int] = {}
        for conv in self.conversations:
            for qa in conv.qa_pairs:
                cat_counts[qa.category] = cat_counts.get(qa.category, 0) + 1
        for cat in ALL_CATEGORIES:
            lines.append(f"  {cat}: {cat_counts.get(cat, 0)}")
        return "\n".join(lines)

    @classmethod
    def load(
        cls,
        path: Optional[str] = None,
        limit: int = 0,
    ) -> "LoCoMoDataset":
        """
        Load the LoCoMo dataset.

        Args:
            path: Path to local ``locomo10.json``. If None, downloads from
                  the snap-research GitHub repo and caches it.
            limit: Maximum number of conversations to load (0 = all).

        Returns:
            Parsed :class:`LoCoMoDataset`.
        """
        if path:
            json_path = Path(path)
        else:
            DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            json_path = DEFAULT_CACHE_DIR / DEFAULT_CACHE_FILE
            if not json_path.exists():
                print(f"  Downloading LoCoMo dataset from {LOCOMO_JSON_URL}...")
                try:
                    urllib.request.urlretrieve(LOCOMO_JSON_URL, str(json_path))
                    print(f"  Cached to {json_path}")
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to download LoCoMo dataset: {e}\n"
                        f"Download manually:\n"
                        f"  curl -fsSL -o {json_path} {LOCOMO_JSON_URL}"
                    ) from e

        if not json_path.exists():
            raise FileNotFoundError(
                f"LoCoMo dataset not found: {json_path}\n"
                f"Download manually:\n"
                f"  curl -fsSL -o {json_path} {LOCOMO_JSON_URL}"
            )

        print(f"  Loading LoCoMo dataset from {json_path.name}...")
        with open(json_path, encoding="utf-8") as f:
            raw = json.load(f)

        conversations = [_parse_conversation(entry) for entry in raw]

        if limit > 0:
            conversations = conversations[:limit]

        dataset = cls(conversations=conversations, source_path=str(json_path))
        print(
            f"  Loaded {len(dataset)} conversations, {dataset.total_qa_pairs} QA pairs"
        )
        return dataset


def _parse_conversation(entry: Dict[str, Any]) -> LoCoMoConversation:
    """Parse a single raw LoCoMo JSON entry into :class:`LoCoMoConversation`."""
    conv_id = str(entry.get("sample_id") or entry.get("conv_id") or "unknown")
    conversation = entry.get("conversation", {}) or {}

    sessions: Dict[int, List[LoCoMoTurn]] = {}
    session_dates: Dict[int, str] = {}

    # Collect session_{N} and session_{N}_date_time keys
    for key, value in conversation.items():
        if key.endswith("_date_time"):
            # session_5_date_time -> 5
            prefix = key[: -len("_date_time")]
            session_num = _extract_session_number(prefix)
            if session_num is not None:
                session_dates[session_num] = str(value)
            continue

        session_num = _extract_session_number(key)
        if session_num is None or not isinstance(value, list):
            continue

        turns: List[LoCoMoTurn] = []
        for idx, raw_turn in enumerate(value):
            if not isinstance(raw_turn, dict):
                continue
            # dia_id in upstream is already "D{session}:{turn}"; fall back to
            # a synthesised ID if the field is missing.
            dia_id = raw_turn.get("dia_id") or f"D{session_num}:{idx}"
            turns.append(
                LoCoMoTurn(
                    turn_id=str(dia_id),
                    session=session_num,
                    turn_index=idx,
                    speaker=str(raw_turn.get("speaker", "")),
                    text=str(raw_turn.get("text", "")),
                )
            )
        sessions[session_num] = turns

    # Attach session date to each turn for convenience
    for session_num, turns in sessions.items():
        date = session_dates.get(session_num, "")
        for t in turns:
            t.date_time = date

    qa_pairs: List[LoCoMoQA] = []
    raw_qa = entry.get("qa", []) or []
    for i, qa in enumerate(raw_qa):
        if not isinstance(qa, dict):
            continue
        cat_id = int(qa.get("category", 0) or 0)
        category = CATEGORY_LABELS.get(cat_id, "unknown")
        evidence = qa.get("evidence") or []
        if not isinstance(evidence, list):
            evidence = []
        qa_pairs.append(
            LoCoMoQA(
                qa_id=f"{conv_id}_qa_{i}",
                question=str(qa.get("question", "")),
                answer=str(qa.get("answer", "")),
                category=category,
                category_id=cat_id,
                evidence_turn_ids=[str(e) for e in evidence],
                adversarial_answer=str(qa.get("adversarial_answer", "")),
            )
        )

    return LoCoMoConversation(
        conv_id=conv_id,
        sessions=sessions,
        session_dates=session_dates,
        qa_pairs=qa_pairs,
    )


def _extract_session_number(key: str) -> Optional[int]:
    """Extract N from a ``session_{N}`` key. Returns None on other keys."""
    if not key.startswith("session_"):
        return None
    suffix = key[len("session_") :]
    # Reject nested keys like session_5_date_time (already stripped by caller)
    if not suffix.isdigit():
        return None
    return int(suffix)
