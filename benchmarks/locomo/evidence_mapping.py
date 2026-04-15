"""
Evidence Mapping

Ingest a :class:`LoCoMoConversation` into an ALMA instance, one turn per
:class:`~alma.types.DomainKnowledge` memory. The turn's LoCoMo evidence
ID (``"D{session}:{turn}"``) is stored in ``metadata["turn_id"]`` so that
retrieval results can be mapped back to the ground-truth evidence IDs.

Returns a ``turn_id -> memory_id`` dict that callers can use to translate
retrieved memory IDs into evidence turn IDs when scoring.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Dict

from alma.types import DomainKnowledge
from benchmarks.locomo.dataset import LoCoMoConversation


def ingest_conversation(
    alma,  # alma.ALMA or anything exposing .storage.save_domain_knowledge
    conv: LoCoMoConversation,
    project_id: str = "locomo",
    agent: str = "benchmark",
    embedder=None,
) -> Dict[str, str]:
    """
    Ingest every turn of ``conv`` as a DomainKnowledge memory.

    Args:
        alma: ALMA instance (or storage-like object) with ``.storage`` exposing
              ``save_domain_knowledge``. If ``alma`` itself has
              ``save_domain_knowledge`` (raw storage), it is used directly.
        conv: Conversation to ingest.
        project_id: Project namespace to store memories under.
        agent: Agent namespace.
        embedder: Optional embedder; if provided, turn embeddings are computed
                  up-front. Otherwise ALMA's default retrieval path will embed
                  on query time (slower but simpler).

    Returns:
        Dict mapping LoCoMo turn IDs (``"D1:3"``) to ALMA memory IDs (UUIDs).
    """
    storage = getattr(alma, "storage", alma)
    if not hasattr(storage, "save_domain_knowledge"):
        raise TypeError(
            "alma argument must expose .storage.save_domain_knowledge or "
            "be a storage instance itself"
        )

    turn_to_memory: Dict[str, str] = {}
    now = datetime.now(timezone.utc)

    for turn in conv.iter_turns():
        text = turn.text.strip()
        if not text:
            continue

        # Prefix speaker so retrieval can surface who said what
        fact_text = f"{turn.speaker}: {text}" if turn.speaker else text

        embedding = embedder.encode(fact_text) if embedder is not None else None

        memory_id = str(uuid.uuid4())
        knowledge = DomainKnowledge(
            id=memory_id,
            agent=agent,
            project_id=project_id,
            domain="conversation",
            fact=fact_text,
            source="locomo",
            confidence=1.0,
            last_verified=now,
            embedding=embedding,
            metadata={
                "turn_id": turn.turn_id,
                "conv_id": conv.conv_id,
                "session": turn.session,
                "turn_index": turn.turn_index,
                "speaker": turn.speaker,
                "date_time": turn.date_time,
            },
        )
        storage.save_domain_knowledge(knowledge)
        turn_to_memory[turn.turn_id] = memory_id

    return turn_to_memory
