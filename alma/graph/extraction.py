"""
ALMA Graph Memory - Entity Extraction.

LLM-powered extraction of entities and relationships from text.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from alma.graph.store import Entity, Relationship

logger = logging.getLogger(__name__)


@dataclass
class ExtractionConfig:
    """Configuration for entity extraction."""
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.1


class EntityExtractor:
    """
    LLM-powered entity and relationship extraction.

    Extracts entities (people, organizations, tools, concepts) and
    relationships between them from text.
    """

    EXTRACTION_PROMPT = '''Extract entities and relationships from the following text.

Entities are things like:
- People (names, roles)
- Organizations (companies, teams)
- Tools/Technologies (software, frameworks)
- Concepts (methodologies, patterns)
- Locations (places, regions)

Relationships describe how entities are connected:
- WORKS_AT (person -> organization)
- USES (entity -> tool)
- KNOWS (person -> person)
- CREATED_BY (thing -> person/org)
- PART_OF (entity -> larger entity)
- RELATED_TO (general relationship)

TEXT:
{text}

Respond in JSON format:
```json
{{
  "entities": [
    {{"id": "unique-id", "name": "Entity Name", "type": "person|organization|tool|concept|location"}}
  ],
  "relationships": [
    {{"source": "entity-id", "target": "entity-id", "type": "RELATIONSHIP_TYPE", "properties": {{}}}}
  ]
}}
```

Only extract entities and relationships that are explicitly mentioned or strongly implied.
'''

    def __init__(self, config: Optional[ExtractionConfig] = None):
        self.config = config or ExtractionConfig()
        self._client = None

    def _get_client(self):
        """Lazy initialization of LLM client."""
        if self._client is None:
            if self.config.provider == "openai":
                from openai import OpenAI
                self._client = OpenAI()
            elif self.config.provider == "anthropic":
                from anthropic import Anthropic
                self._client = Anthropic()
        return self._client

    def extract(
        self,
        text: str,
        existing_entities: Optional[List[Entity]] = None,
    ) -> Tuple[List[Entity], List[Relationship]]:
        """
        Extract entities and relationships from text.

        Args:
            text: Text to extract from
            existing_entities: Optional list of known entities for linking

        Returns:
            Tuple of (entities, relationships)
        """
        import json
        import re
        import uuid

        prompt = self.EXTRACTION_PROMPT.format(text=text)

        client = self._get_client()

        if self.config.provider == "openai":
            response = client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
            )
            raw_response = response.choices[0].message.content
        elif self.config.provider == "anthropic":
            response = client.messages.create(
                model=self.config.model,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
            )
            raw_response = response.content[0].text
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")

        # Parse JSON from response
        json_match = re.search(r'```json\s*(.*?)\s*```', raw_response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
            json_str = json_match.group(0) if json_match else "{}"

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse extraction response: {raw_response[:200]}")
            return [], []

        # Build entities
        entities = []
        entity_map = {}  # id -> Entity

        for e in data.get("entities", []):
            entity_id = e.get("id") or str(uuid.uuid4())[:8]
            entity = Entity(
                id=entity_id,
                name=e.get("name", "Unknown"),
                entity_type=e.get("type", "concept"),
            )
            entities.append(entity)
            entity_map[entity_id] = entity

            # Also map by name for relationship linking
            entity_map[e.get("name", "").lower()] = entity

        # Build relationships
        relationships = []
        for r in data.get("relationships", []):
            source_id = r.get("source")
            target_id = r.get("target")

            # Try to resolve IDs
            source = entity_map.get(source_id) or entity_map.get(source_id.lower() if source_id else "")
            target = entity_map.get(target_id) or entity_map.get(target_id.lower() if target_id else "")

            if source and target:
                rel = Relationship(
                    id=f"{source.id}-{r.get('type', 'RELATED')}-{target.id}",
                    source_id=source.id,
                    target_id=target.id,
                    relation_type=r.get("type", "RELATED_TO"),
                    properties=r.get("properties", {}),
                )
                relationships.append(rel)

        return entities, relationships

    def extract_from_conversation(
        self,
        messages: List[Dict[str, str]],
    ) -> Tuple[List[Entity], List[Relationship]]:
        """
        Extract entities and relationships from a conversation.

        Args:
            messages: List of {"role": "...", "content": "..."}

        Returns:
            Tuple of (entities, relationships)
        """
        # Combine messages into text
        text = "\n".join(
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in messages
        )

        return self.extract(text)
