"""
ALMA Extraction Module.

LLM-powered and rule-based fact extraction from conversations,
plus regex-based entity detection from text.
"""

from alma.extraction.auto_learner import (
    AutoLearner,
    add_auto_learning_to_alma,
)
from alma.extraction.entity_detector import (
    detect_entities,
    detect_entities_from_file,
)
from alma.extraction.extractor import (
    ExtractedFact,
    ExtractionResult,
    FactExtractor,
    FactType,
    LLMFactExtractor,
    RuleBasedExtractor,
    create_extractor,
)

__all__ = [
    "FactExtractor",
    "LLMFactExtractor",
    "RuleBasedExtractor",
    "ExtractedFact",
    "ExtractionResult",
    "FactType",
    "create_extractor",
    "AutoLearner",
    "add_auto_learning_to_alma",
    "detect_entities",
    "detect_entities_from_file",
]
