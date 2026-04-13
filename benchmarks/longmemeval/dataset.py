"""
LongMemEval Dataset Loader

Loads the LongMemEval benchmark dataset from HuggingFace or a local JSON file.

Dataset structure (per entry):
    - question_id: str -- unique question identifier
    - question: str -- the natural language question
    - question_type: str -- category (e.g. "single-session-user", "temporal-reasoning")
    - question_date: str -- date context for the question (format: "2023/01/15 (Sun) 10:20")
    - answer: str -- ground-truth answer text
    - answer_session_ids: List[str] -- session IDs containing the answer
    - haystack_sessions: List[List[Dict]] -- conversation sessions, each a list of
      {"role": "user"|"assistant", "content": str} turns
    - haystack_session_ids: List[str] -- parallel list of session IDs
    - haystack_dates: List[str] -- parallel list of session dates

HuggingFace dataset: xiaowu0162/longmemeval-cleaned
Direct JSON: longmemeval_s_cleaned.json
"""

import json
import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional  # noqa: I001

# HuggingFace direct download URL for the cleaned dataset
HUGGINGFACE_URL = (
    "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned"
    "/resolve/main/longmemeval_s_cleaned.json"
)

# Default local cache location
DEFAULT_CACHE_DIR = Path(
    os.environ.get("ALMA_BENCHMARK_DATA", "/tmp/alma-benchmark-data")
)
DEFAULT_CACHE_FILE = "longmemeval_s_cleaned.json"


@dataclass
class ConversationTurn:
    """A single turn in a conversation session."""

    role: str  # "user" or "assistant"
    content: str


@dataclass
class Session:
    """A conversation session from the haystack."""

    session_id: str
    date: str
    turns: List[ConversationTurn]

    def user_text(self) -> str:
        """Join all user turns into a single text block."""
        return "\n".join(t.content for t in self.turns if t.role == "user")

    def full_text(self) -> str:
        """Join all turns (user + assistant) into a single text block."""
        return "\n".join(t.content for t in self.turns)


@dataclass
class BenchmarkQuestion:
    """A single benchmark question with its context and ground truth."""

    question_id: str
    question: str
    question_type: str
    question_date: str
    answer: str
    answer_session_ids: List[str]
    haystack: List[Session]

    @property
    def correct_session_ids(self) -> set:
        """Set of session IDs that contain the correct answer."""
        return set(self.answer_session_ids)


@dataclass
class LongMemEvalDataset:
    """
    The complete LongMemEval benchmark dataset.

    Contains 500 questions across 5 ability categories, each with
    a haystack of ~53 conversation sessions to search through.
    """

    questions: List[BenchmarkQuestion] = field(default_factory=list)
    source_path: str = ""

    def __len__(self) -> int:
        return len(self.questions)

    def __getitem__(self, idx: int) -> BenchmarkQuestion:
        return self.questions[idx]

    def filter_by_type(self, question_type: str) -> "LongMemEvalDataset":
        """Return a new dataset containing only questions of the given type."""
        filtered = [q for q in self.questions if q.question_type == question_type]
        return LongMemEvalDataset(questions=filtered, source_path=self.source_path)

    @property
    def question_types(self) -> List[str]:
        """Return sorted list of unique question types in the dataset."""
        return sorted({q.question_type for q in self.questions})

    def summary(self) -> str:
        """Return a human-readable summary of the dataset."""
        lines = [f"LongMemEval Dataset: {len(self)} questions"]
        type_counts: Dict[str, int] = {}
        for q in self.questions:
            type_counts[q.question_type] = type_counts.get(q.question_type, 0) + 1
        for qt, count in sorted(type_counts.items()):
            lines.append(f"  {qt}: {count}")
        return "\n".join(lines)


def _parse_entry(entry: Dict[str, Any]) -> BenchmarkQuestion:
    """
    Parse a raw JSON entry into a BenchmarkQuestion.

    Args:
        entry: Raw dictionary from the JSON dataset

    Returns:
        Parsed BenchmarkQuestion with structured session data
    """
    sessions = []
    raw_sessions = entry.get("haystack_sessions", [])
    session_ids = entry.get("haystack_session_ids", [])
    dates = entry.get("haystack_dates", [])

    for raw_session, sess_id, date in zip(
        raw_sessions, session_ids, dates, strict=False
    ):
        turns = [
            ConversationTurn(role=t["role"], content=t["content"]) for t in raw_session
        ]
        sessions.append(Session(session_id=sess_id, date=date, turns=turns))

    return BenchmarkQuestion(
        question_id=entry["question_id"],
        question=entry["question"],
        question_type=entry.get("question_type", "unknown"),
        question_date=entry.get("question_date", ""),
        answer=entry.get("answer", ""),
        answer_session_ids=entry.get("answer_session_ids", []),
        haystack=sessions,
    )


def load_from_file(path: str) -> LongMemEvalDataset:
    """
    Load dataset from a local JSON file.

    Args:
        path: Path to the longmemeval_s_cleaned.json file

    Returns:
        Parsed LongMemEvalDataset

    Raises:
        FileNotFoundError: If the file does not exist
        json.JSONDecodeError: If the file is not valid JSON
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}\n"
            f"Download it first:\n"
            f"  curl -fsSL -o {path} {HUGGINGFACE_URL}"
        )

    print(f"  Loading dataset from {file_path.name}...")
    with open(file_path, encoding="utf-8") as f:
        raw_data = json.load(f)

    questions = [_parse_entry(entry) for entry in raw_data]
    dataset = LongMemEvalDataset(questions=questions, source_path=str(path))
    print(f"  Loaded {len(dataset)} questions")
    return dataset


def load_from_huggingface(
    cache_dir: Optional[Path] = None,
) -> LongMemEvalDataset:
    """
    Load dataset from HuggingFace, using the datasets library if available,
    otherwise falling back to direct JSON download.

    Args:
        cache_dir: Directory to cache the downloaded file. Defaults to
                   ALMA_BENCHMARK_DATA env var or /tmp/alma-benchmark-data.

    Returns:
        Parsed LongMemEvalDataset
    """
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / DEFAULT_CACHE_FILE

    # If cached file exists, load from it
    if cache_file.exists():
        return load_from_file(str(cache_file))

    # Try datasets library first
    try:
        import datasets

        print("  Loading from HuggingFace using datasets library...")
        ds = datasets.load_dataset("xiaowu0162/longmemeval-cleaned", split="train")
        # Convert to our format and save as JSON for future use
        raw_data = [dict(entry) for entry in ds]
        with open(cache_file, "w") as f:
            json.dump(raw_data, f)
        print(f"  Cached to {cache_file}")
        return load_from_file(str(cache_file))
    except ImportError:
        pass

    # Fallback: direct HTTP download
    print("  Downloading dataset from HuggingFace...")
    print(f"  URL: {HUGGINGFACE_URL}")
    try:
        urllib.request.urlretrieve(HUGGINGFACE_URL, str(cache_file))
        print(f"  Downloaded to {cache_file}")
        return load_from_file(str(cache_file))
    except Exception as e:
        raise RuntimeError(
            f"Failed to download dataset: {e}\n"
            f"Download manually:\n"
            f"  curl -fsSL -o {cache_file} {HUGGINGFACE_URL}"
        ) from e


def load_dataset(
    path: Optional[str] = None,
    limit: int = 0,
    skip: int = 0,
) -> LongMemEvalDataset:
    """
    Load the LongMemEval dataset.

    Unified entry point that handles local files and HuggingFace download.

    Args:
        path: Path to local JSON file. If None, downloads from HuggingFace.
        limit: Maximum number of questions to load (0 = all)
        skip: Number of questions to skip from the start

    Returns:
        Parsed LongMemEvalDataset, optionally truncated
    """
    if path:
        dataset = load_from_file(path)
    else:
        dataset = load_from_huggingface()

    questions = dataset.questions

    if skip > 0:
        questions = questions[skip:]

    if limit > 0:
        questions = questions[:limit]

    return LongMemEvalDataset(
        questions=questions,
        source_path=dataset.source_path,
    )
