# Installation

## Python

### Basic Installation

```bash
pip install alma-memory
```

### With Optional Backends

```bash
# Local development (SQLite + FAISS + local embeddings)
pip install alma-memory[local]

# Production databases
pip install alma-memory[postgres]   # PostgreSQL + pgvector
pip install alma-memory[qdrant]     # Qdrant vector database
pip install alma-memory[pinecone]   # Pinecone vector database
pip install alma-memory[chroma]     # ChromaDB

# Enterprise
pip install alma-memory[azure]      # Azure Cosmos DB + Azure OpenAI

# Everything
pip install alma-memory[all]
```

### Requirements

- Python 3.10 or higher
- pip 21.0 or higher

## TypeScript / JavaScript

### Via GitHub Packages

```bash
# Configure npm for the scope (one-time)
echo "@rbkunnela:registry=https://npm.pkg.github.com" >> ~/.npmrc

# Install
npm install @rbkunnela/alma-memory
```

Or with yarn:

```bash
yarn add @rbkunnela/alma-memory
```

### Requirements

- Node.js 18.0.0 or higher (for native `fetch` support)
- A running ALMA MCP server for the TypeScript SDK to connect to

## Verify Installation

=== "Python"

    ```python
    from alma import ALMA, __version__
    print(f"ALMA version: {__version__}")
    ```

=== "TypeScript"

    ```typescript
    import { ALMA, VERSION } from '@rbkunnela/alma-memory';
    console.log(`ALMA version: ${VERSION}`);
    ```

## Next Steps

- [Quick Start Guide](quickstart.md) - Build your first memory-powered agent
- [Configuration](configuration.md) - Set up storage backends and agent scopes
