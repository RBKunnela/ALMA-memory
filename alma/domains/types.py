"""
Domain Memory Types.

Data models for domain-specific memory schemas.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import uuid


@dataclass
class EntityType:
    """
    A type of entity in a domain.

    Entities are the "things" that agents work with in a domain.
    For example: features, bugs, tests (coding), papers, hypotheses (research).
    """

    name: str  # "feature", "test", "paper", "lead"
    description: str
    attributes: List[str] = field(default_factory=list)  # ["status", "priority", "owner"]

    # Optional schema validation
    required_attributes: List[str] = field(default_factory=list)
    attribute_types: Dict[str, str] = field(default_factory=dict)  # attr -> "str", "int", "bool"

    def validate_entity(self, entity: Dict[str, Any]) -> List[str]:
        """Validate an entity instance against this type."""
        errors = []
        for attr in self.required_attributes:
            if attr not in entity:
                errors.append(f"Missing required attribute: {attr}")
        return errors


@dataclass
class RelationshipType:
    """
    A relationship between entities in a domain.

    Relationships connect entities (e.g., "feature implements spec").
    """

    name: str  # "implements", "blocks", "supports", "cites"
    description: str
    source_type: str  # Entity type name
    target_type: str  # Entity type name

    # Cardinality
    many_to_many: bool = True
    required: bool = False


@dataclass
class DomainSchema:
    """
    Defines memory structure for a specific domain.

    A schema describes what entities exist in a domain, how they relate,
    and what learning categories agents can use.
    """

    id: str
    name: str  # "coding", "research", "sales"
    description: str

    # What entities exist in this domain
    entity_types: List[EntityType] = field(default_factory=list)

    # What relationships between entities
    relationship_types: List[RelationshipType] = field(default_factory=list)

    # Learning categories (replaces hardcoded HELENA_CATEGORIES)
    learning_categories: List[str] = field(default_factory=list)

    # What can agents in this domain NOT learn (scoping)
    excluded_categories: List[str] = field(default_factory=list)

    # Domain-specific settings
    min_occurrences_for_heuristic: int = 3
    confidence_decay_days: float = 30.0

    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Extensible metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        name: str,
        description: str,
        learning_categories: Optional[List[str]] = None,
        **kwargs,
    ) -> "DomainSchema":
        """Factory method to create a new domain schema."""
        return cls(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            learning_categories=learning_categories or [],
            **kwargs,
        )

    def add_entity_type(
        self,
        name: str,
        description: str,
        attributes: Optional[List[str]] = None,
    ) -> EntityType:
        """Add an entity type to this schema."""
        entity = EntityType(
            name=name,
            description=description,
            attributes=attributes or [],
        )
        self.entity_types.append(entity)
        self.updated_at = datetime.now(timezone.utc)
        return entity

    def add_relationship_type(
        self,
        name: str,
        description: str,
        source_type: str,
        target_type: str,
    ) -> RelationshipType:
        """Add a relationship type to this schema."""
        rel = RelationshipType(
            name=name,
            description=description,
            source_type=source_type,
            target_type=target_type,
        )
        self.relationship_types.append(rel)
        self.updated_at = datetime.now(timezone.utc)
        return rel

    def add_learning_category(self, category: str) -> None:
        """Add a learning category."""
        if category not in self.learning_categories:
            self.learning_categories.append(category)
            self.updated_at = datetime.now(timezone.utc)

    def add_excluded_category(self, category: str) -> None:
        """Add an excluded category (agent cannot learn from this)."""
        if category not in self.excluded_categories:
            self.excluded_categories.append(category)
            self.updated_at = datetime.now(timezone.utc)

    def get_entity_type(self, name: str) -> Optional[EntityType]:
        """Get entity type by name."""
        for entity in self.entity_types:
            if entity.name == name:
                return entity
        return None

    def get_relationship_type(self, name: str) -> Optional[RelationshipType]:
        """Get relationship type by name."""
        for rel in self.relationship_types:
            if rel.name == name:
                return rel
        return None

    def is_category_allowed(self, category: str) -> bool:
        """Check if a learning category is allowed in this domain."""
        if self.learning_categories and category not in self.learning_categories:
            return False
        if category in self.excluded_categories:
            return False
        return True

    def validate(self) -> List[str]:
        """Validate the schema for consistency."""
        errors = []

        # Check relationship source/target types exist
        entity_names = {e.name for e in self.entity_types}
        for rel in self.relationship_types:
            if rel.source_type not in entity_names:
                errors.append(
                    f"Relationship '{rel.name}' references unknown source type: {rel.source_type}"
                )
            if rel.target_type not in entity_names:
                errors.append(
                    f"Relationship '{rel.name}' references unknown target type: {rel.target_type}"
                )

        # Check for duplicate entity names
        seen_names = set()
        for entity in self.entity_types:
            if entity.name in seen_names:
                errors.append(f"Duplicate entity type name: {entity.name}")
            seen_names.add(entity.name)

        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert schema to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "entity_types": [
                {
                    "name": e.name,
                    "description": e.description,
                    "attributes": e.attributes,
                }
                for e in self.entity_types
            ],
            "relationship_types": [
                {
                    "name": r.name,
                    "description": r.description,
                    "source_type": r.source_type,
                    "target_type": r.target_type,
                }
                for r in self.relationship_types
            ],
            "learning_categories": self.learning_categories,
            "excluded_categories": self.excluded_categories,
            "min_occurrences_for_heuristic": self.min_occurrences_for_heuristic,
            "confidence_decay_days": self.confidence_decay_days,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DomainSchema":
        """Create schema from dictionary."""
        entity_types = [
            EntityType(
                name=e["name"],
                description=e["description"],
                attributes=e.get("attributes", []),
            )
            for e in data.get("entity_types", [])
        ]

        relationship_types = [
            RelationshipType(
                name=r["name"],
                description=r["description"],
                source_type=r["source_type"],
                target_type=r["target_type"],
            )
            for r in data.get("relationship_types", [])
        ]

        return cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data["name"],
            description=data["description"],
            entity_types=entity_types,
            relationship_types=relationship_types,
            learning_categories=data.get("learning_categories", []),
            excluded_categories=data.get("excluded_categories", []),
            min_occurrences_for_heuristic=data.get("min_occurrences_for_heuristic", 3),
            confidence_decay_days=data.get("confidence_decay_days", 30.0),
            metadata=data.get("metadata", {}),
        )
