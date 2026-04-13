# Technical Reference - Code Patterns from MemPalace

Copy these patterns. Don't reinvent.

---

## Pattern 1: ChromaDB Collection Access (from mempalace/palace.py)

### What MemPalace Does
```python
def get_collection(palace_path: str, collection_name: str = "mempalace_drawers"):
    """Get ChromaDB collection with proper setup"""
    client = chromadb.PersistentClient(path=palace_path)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    return collection
```

### How to Adapt for ALMA
```python
# alma/storage/chroma.py

class ChromaBackend:
    def __init__(self, path: str = ".alma/chroma"):
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(
            name="alma_memories",
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_memory(self, memory: Memory):
        """Add to ChromaDB matching MemPalace pattern"""
        self.collection.add(
            ids=[memory.id],
            embeddings=[memory.embedding],
            documents=[memory.content],
            metadatas=[{
                "wing": memory.wing,
                "hall": memory.hall,
                "room": memory.room,
                "agent_id": memory.agent_id,
                "memory_type": memory.memory_type,
                "source_file": memory.source_file,
                "source_mtime": memory.source_mtime,
                **memory.metadata,  # flatten metadata dict
            }]
        )
    
    def query(self, query_embedding, where=None, n_results=5):
        """Query matching MemPalace's interface"""
        return self.collection.query(
            query_embeddings=[query_embedding],
            where=where,
            n_results=n_results,
            include=["embeddings", "documents", "metadatas", "distances"]
        )
```

---

## Pattern 2: Embedding (from mempalace/cli.py)

### What MemPalace Uses
```python
from sentence_transformers import SentenceTransformer

# Load once at startup
_embedder = SentenceTransformer("all-MiniLM-L6-v2")

def embed(text: str) -> List[float]:
    """Embed text - 384 dims, runs locally"""
    return _embedder.encode(text, convert_to_numpy=False)
```

### How to Adapt
```python
# alma/core/embedding.py

from sentence_transformers import SentenceTransformer
from functools import lru_cache

@lru_cache(maxsize=1)
def get_embedder():
    """Lazy-load embedder once"""
    return SentenceTransformer("all-MiniLM-L6-v2")

def embed(text: str):
    """Embed with caching"""
    embedder = get_embedder()
    return embedder.encode(text, convert_to_numpy=False).tolist()
```

---

## Pattern 3: Configuration (from mempalace/config.py)

### What MemPalace Does
```python
from dataclasses import dataclass
import yaml
from pathlib import Path

@dataclass
class MempalaceConfig:
    palace_path: str
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    
    @classmethod
    def from_yaml(cls, path: str):
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
```

### How to Adapt
```python
# alma/config.py

@dataclass
class AlmaConfig:
    # Paths
    project_id: str
    storage_dir: str = ".alma"
    db_path: str = ".alma/alma.db"
    
    # Storage
    storage_type: str = "sqlite"  # sqlite|postgres|qdrant|pinecone|chroma
    
    # Embedding
    embedding_provider: str = "local"  # local|azure|openai
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    
    # Agents (can be empty for MVP)
    agents: Dict[str, Dict] = None
    
    @classmethod
    def from_yaml(cls, path: str):
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data.get("alma", {}))
    
    @classmethod
    def from_env(cls):
        """Load from environment variables"""
        return cls(
            project_id=os.getenv("ALMA_PROJECT_ID", "default"),
            storage_dir=os.getenv("ALMA_STORAGE_DIR", ".alma"),
        )
```

---

## Pattern 4: Metadata Filtering (from mempalace/searcher.py)

### What MemPalace Does
```python
def search_memories(query, wing=None, hall=None, room=None):
    where = {}
    if wing:
        where["wing"] = {"$eq": wing}
    if hall:
        where["hall"] = {"$eq": hall}
    if room:
        where["room"] = {"$eq": room}
    
    results = collection.query(
        query_embeddings=[embed(query)],
        where=where if where else None,
        n_results=5
    )
    return results
```

### How to Adapt (ALMA with scoping)
```python
# alma/core/search/searcher.py

def build_where_clause(agent_id: str, agent: Agent, wing=None):
    """Build Chroma where clause with agent scoping"""
    conditions = []
    
    # Agent must own or inherit this memory
    conditions.append({
        "$or": [
            {"agent_id": {"$eq": agent_id}},
            {"agent_id": {"$in": agent.inherit_from}},  # can inherit
        ]
    })
    
    # Can't learn in these categories
    if agent.cannot_learn:
        conditions.append({
            "memory_type": {"$nin": agent.cannot_learn}
        })
    
    # Can only learn in these categories (if list is non-empty)
    if agent.can_learn:
        conditions.append({
            "memory_type": {"$in": agent.can_learn}
        })
    
    # Palace navigation (optional)
    if wing:
        conditions.append({"wing": {"$eq": wing}})
    
    return {"$and": conditions} if conditions else None
```

---

## Pattern 5: Query Sanitization (from mempalace/query_sanitizer.py)

### What MemPalace Does
```python
def sanitize_query(query: str, max_length: int = 1000) -> str:
    """Clean user input"""
    # Remove control characters
    query = "".join(c for c in query if ord(c) >= 32 or c in "\n\t")
    # Truncate
    query = query[:max_length]
    # Strip whitespace
    return query.strip()
```

### Use As-Is in ALMA
```python
# alma/core/search/query_sanitizer.py - copy MemPalace's implementation
```

---

## Pattern 6: Entity Detection (from mempalace/entity_detector.py)

### What MemPalace Does
```python
import spacy

def detect_entities(text: str):
    """Use SpaCy for NER"""
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    
    entities = {}
    for ent in doc.ents:
        entity_type = ent.label_
        if entity_type not in entities:
            entities[entity_type] = []
        entities[entity_type].append(ent.text)
    
    return entities
```

### Adapt for ALMA
```python
# alma/core/mining/entity_detector.py - can copy MemPalace's pattern
# Use for: extracting "room" names from task descriptions
```

---

## Pattern 7: Deduplication (from mempalace/dedup.py)

### What MemPalace Does
```python
def is_duplicate(content1: str, content2: str, threshold: float = 0.9):
    """Check if two pieces of content are duplicates"""
    embed1 = embed(content1)
    embed2 = embed(content2)
    
    similarity = np.dot(embed1, embed2) / (
        np.linalg.norm(embed1) * np.linalg.norm(embed2)
    )
    
    return similarity > threshold
```

### Adapt for ALMA Consolidation
```python
# alma/learning/consolidation.py

def find_duplicate_memories(agent_id: str, threshold: float = 0.85):
    """Find similar memories for consolidation"""
    # Get all memories for this agent
    memories = backend.get_memories(agent_id=agent_id)
    
    # Build similarity matrix
    duplicates = []
    for i, mem1 in enumerate(memories):
        for j, mem2 in enumerate(memories[i+1:], start=i+1):
            similarity = cosine_similarity(mem1.embedding, mem2.embedding)
            if similarity > threshold:
                duplicates.append((mem1.id, mem2.id, similarity))
    
    return duplicates
```

---

## Pattern 8: Conversation Mining (from mempalace/convo_miner.py)

### What MemPalace Does
```python
def mine_conversations(export_file: str):
    """Extract memories from ChatGPT/Claude export"""
    with open(export_file) as f:
        data = json.load(f)
    
    memories = []
    for conversation in data:
        for message in conversation["messages"]:
            if message["role"] == "assistant":
                # Extract key facts, decisions, advice
                memories.append({
                    "content": message["content"],
                    "hall": "facts",
                })
    
    return memories
```

### How to Adapt
```python
# alma/core/mining/convo_miner.py

def mine_conversations(export_file: str, agent_id: str):
    """Mine with agent assignment"""
    with open(export_file) as f:
        data = json.load(f)
    
    memories = []
    for conversation in data:
        for message in conversation["messages"]:
            if message["role"] == "assistant":
                memory = Memory(
                    id=str(uuid4()),
                    wing=agent_id,
                    hall="facts",
                    room=extract_topic(conversation),  # infer from first message
                    agent_id=agent_id,
                    memory_type="domain_knowledge",
                    content=message["content"],
                )
                memories.append(memory)
    
    return memories
```

---

## Pattern 9: Testing Fixtures (from mempalace/tests/conftest.py)

### What MemPalace Does
```python
import pytest
from pathlib import Path
import tempfile

@pytest.fixture
def test_palace():
    """Temporary palace for testing"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

@pytest.fixture
def test_collection(test_palace):
    """Get test ChromaDB collection"""
    from mempalace.palace import get_collection
    return get_collection(test_palace)

@pytest.fixture
def sample_drawer():
    """Sample drawer for testing"""
    return {
        "id": "test/facts/sample/uuid",
        "content": "Test content",
        "metadata": {
            "wing": "test",
            "hall": "facts",
            "room": "sample",
        }
    }
```

### Adapt for ALMA
```python
# tests/conftest.py

@pytest.fixture
def test_alma():
    """Temporary ALMA instance"""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = AlmaConfig(
            project_id="test",
            storage_dir=tmpdir,
            db_path=f"{tmpdir}/test.db"
        )
        alma = ALMA(config)
        yield alma

@pytest.fixture
def test_agent(test_alma):
    """Create test agent"""
    agent = Agent(
        id="test_agent",
        name="Test Agent",
        domain="coding",
        can_learn=["testing_strategies", "code_patterns"],
        cannot_learn=[],
    )
    test_alma.agent_manager.store_agent(agent)
    return agent

@pytest.fixture
def sample_memory():
    """Sample memory for testing"""
    return Memory(
        id="test/facts/sample/uuid",
        wing="test",
        hall="facts",
        room="sample",
        agent_id="test_agent",
        memory_type="domain_knowledge",
        content="Test content",
        embedding=[0.1] * 384,
    )
```

---

## Pattern 10: MCP Server (from mempalace/mcp_server.py)

### What MemPalace Does
```python
import anthropic
from mcp import types

server = Server("mempalace")

@server.list_tools()
async def list_tools():
    return [
        types.Tool(
            name="mempalace_search",
            description="Search memories",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "wing": {"type": "string"},
                },
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "mempalace_search":
        results = search_memories(arguments["query"], wing=arguments.get("wing"))
        return [types.TextContent(text=json.dumps(results))]
```

### Adapt for ALMA (Sprint 6)
```python
# alma/integration/mcp_server.py - copy MemPalace's pattern, add:
# - alma_retrieve (with agent_id + scoping)
# - alma_learn (with validation)
# - alma_manage_agent (create/update agents)
```

---

## Pattern 11: Benchmark Structure (from mempalace/benchmarks/)

### What MemPalace Does
```python
# benchmarks/longmemeval_bench.py

def run_benchmark():
    dataset = load_longmemeval()
    scores = []
    
    for question in dataset:
        results = search_memories(question["query"])
        is_correct = check_answer(results, question["expected"])
        scores.append(is_correct)
    
    accuracy = sum(scores) / len(scores)
    print(f"LongMemEval Accuracy: {accuracy:.1%}")
```

### Adapt for ALMA
```python
# benchmarks/longmemeval_with_scoping.py

def run_benchmark_with_scoping():
    dataset = load_longmemeval()
    
    # Create test agents with different scopes
    agents = create_test_agents()
    
    scores = {"baseline": [], "with_scoping": []}
    
    for question in dataset:
        # Baseline: no scoping
        results_baseline = search_memories_baseline(question["query"])
        
        # With scoping
        results_scoped = search_memories(
            question["query"],
            agent_id="test_agent",  # with boundaries
        )
        
        scores["baseline"].append(check_answer(results_baseline, question["expected"]))
        scores["with_scoping"].append(check_answer(results_scoped, question["expected"]))
    
    print(f"Baseline: {sum(scores['baseline'])/len(scores['baseline']):.1%}")
    print(f"With scoping: {sum(scores['with_scoping'])/len(scores['with_scoping']):.1%}")
```

---

## What Files to Directly Copy from MemPalace

These don't need adaptation - use as-is:

```
mempalace/query_sanitizer.py         → alma/core/search/query_sanitizer.py
mempalace/spellcheck.py              → alma/core/mining/spellcheck.py
mempalace/normalize.py               → alma/core/mining/normalize.py
mempalace/dedup.py                   → alma/core/mining/dedup.py (light adapt)
mempalace/entity_detector.py         → alma/core/mining/entity_detector.py (light adapt)
mempalace/convo_miner.py             → alma/core/mining/convo_miner.py (light adapt)
tests/test_dedup.py                  → tests/test_consolidation.py (adapt)
tests/test_entity_detector.py        → tests/test_entity_detection.py (adapt)
```

---

## What to Adapt from MemPalace

These need ALMA-specific changes:

```
mempalace/backends/chroma.py         → alma/storage/chroma.py (add to schema)
mempalace/palace.py                  → alma/core/memory.py (simplify)
mempalace/searcher.py                → alma/core/search/searcher.py (add scoping)
mempalace/mcp_server.py              → alma/integration/mcp_server.py (Sprint 6)
mempalace/cli.py                     → alma/cli.py (add governance commands)
```

---

## What NOT to Copy

These are MemPalace-specific, ALMA doesn't need them:

```
mempalace/dialect.py                 # AAAK compression (skip for MVP)
mempalace/split_mega_files.py        # Edge case (skip for MVP)
mempalace/room_detector_local.py     # Room auto-detection (skip for MVP)
mempalace/knowledge_graph.py         # Graph not needed for MVP
mempalace/layers.py                  # L0/L1 caching (add later)
mempalace/migrate.py                 # MemPalace's internal migrations
mempalace/repair.py                  # Database repair (add after MVP)
```

---

## Quick Import Checklist

Make sure you have these packages in `alma/pyproject.toml`:

```toml
[project]
dependencies = [
    "chromadb>=0.4.0",                    # Vector DB
    "sentence-transformers>=2.2.0",       # Embedding model
    "pydantic>=2.0.0",                    # Validation
    "pyyaml>=6.0",                        # Config files
    "numpy>=1.23.0",                      # Math
    "python-dotenv>=1.0.0",               # Env vars
    "anthropic>=0.21.0",                  # Claude API (for optional LLM classifier)
    "spacy>=3.5.0",                       # NLP (optional, for mining)
]

[project.optional-dependencies]
postgres = ["psycopg>=3.1.0", "pgvector>=0.1.0"]
qdrant = ["qdrant-client>=2.5.0"]
pinecone = ["pinecone>=3.0.0"]
azure = ["azure-cosmos>=4.5.0"]
all = ["psycopg>=3.1.0", "pgvector>=0.1.0", "qdrant-client>=2.5.0", "pinecone>=3.0.0", "azure-cosmos>=4.5.0"]
dev = ["pytest>=7.0.0", "pytest-asyncio>=0.21.0", "black>=23.0.0", "ruff>=0.1.0"]
```

---

## Testing Strategy from MemPalace

MemPalace has 11K lines of tests. Follow the same pattern:

1. **Unit tests** - test individual functions
   - `test_embedding.py` - embed() function
   - `test_schema.py` - Memory/Agent dataclasses
   - `test_sanitizer.py` - query sanitization

2. **Integration tests** - test components together
   - `test_retrieval.py` - search + ChromaDB
   - `test_learning.py` - learn + storage + validation

3. **End-to-end tests** - full pipeline
   - `test_e2e_alma_mempalace.py` - migrate + search + learn

4. **Benchmark tests** - measure quality
   - `test_longmemeval.py` - retrieval accuracy
   - `test_consolidation.py` - dedup quality

Target: 1:1 test-to-code ratio (MemPalace: 11.5K code, 11.3K tests)

---

## Key Takeaway

Don't redesign. Copy MemPalace's proven patterns, add ALMA's governance on top, measure ruthlessly.

If MemPalace does it a certain way and it works at 96%+ accuracy, your job is to integrate ALMA's scoping without breaking that accuracy.

Use these patterns. They're battle-tested.
