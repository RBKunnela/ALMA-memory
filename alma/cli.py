"""
ALMA CLI — alma init, alma doctor, alma test, alma pg-setup

Usage:
    alma init                    # Interactive setup wizard
    alma init --quickstart       # Zero-config SQLite in 10 seconds
    alma init --storage postgres # Guide through PostgreSQL setup
    alma doctor                  # Check install health
    alma test                    # Run a live store+retrieve test
    alma pg-setup                # Print or run PostgreSQL setup SQL
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Suppress noisy warnings from HuggingFace, tokenizers, and sentence-transformers
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("HUGGINGFACE_HUB_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import warnings as _warnings
_warnings.filterwarnings("ignore", category=FutureWarning, module="sentence_transformers")
_warnings.filterwarnings("ignore", category=DeprecationWarning, module="sentence_transformers")


# ─── helpers ──────────────────────────────────────────────────────────────────

def _green(s: str) -> str:
    return f"\033[32m{s}\033[0m"

def _yellow(s: str) -> str:
    return f"\033[33m{s}\033[0m"

def _red(s: str) -> str:
    return f"\033[31m{s}\033[0m"

def _bold(s: str) -> str:
    return f"\033[1m{s}\033[0m"

def _check(label: str, ok: bool, fix: str = "") -> bool:
    status = _green("✓") if ok else _red("✗")
    print(f"  {status}  {label}")
    if not ok and fix:
        print(f"     {_yellow('→')} {fix}")
    return ok


# ─── init ─────────────────────────────────────────────────────────────────────

QUICKSTART_CONFIG = """\
alma:
  project_id: "{project_id}"
  storage: sqlite
  embedding_provider: local
  embedding_dim: 384
  storage_dir: {alma_dir}
  db_name: alma.db
"""

POSTGRES_CONFIG = """\
alma:
  project_id: "{project_id}"
  storage: postgres
  embedding_provider: local
  embedding_dim: 384

  postgres:
    host: {host}
    port: {port}
    database: {database}
    user: {user}
    password: ${{ALMA_DB_PASSWORD}}
    vector_index_type: hnsw
"""


def cmd_init(args: argparse.Namespace) -> int:
    alma_dir = Path(args.dir).expanduser().resolve()
    config_path = alma_dir / "config.yaml"

    print(_bold("\nALMA Setup\n"))

    if config_path.exists() and not args.force:
        print(f"  Config already exists: {config_path}")
        print(f"  Use --force to overwrite.\n")
        return 0

    alma_dir.mkdir(parents=True, exist_ok=True)

    # ── quickstart: zero questions, SQLite ────────────────────────────────────
    if args.quickstart or args.storage == "sqlite":
        project_id = args.project or Path.cwd().name or "my-project"
        config_path.write_text(
            QUICKSTART_CONFIG.format(project_id=project_id, alma_dir=str(alma_dir))
        )
        print(f"  {_green('✓')}  Config written: {config_path}")
        print(f"  {_green('✓')}  Storage:        SQLite ({alma_dir}/alma.db)")
        print(f"  {_green('✓')}  Embeddings:     local (all-MiniLM-L6-v2)\n")

        # Check if sentence-transformers is installed
        try:
            import sentence_transformers  # noqa: F401
            print(_green("  Ready. Run: alma test\n"))
        except ImportError:
            print(_yellow("  One more step — install local embeddings:"))
            print(f"    pip install 'alma-memory[local]'\n")
            print("  Then run: alma test\n")
        return 0

    # ── postgres setup ────────────────────────────────────────────────────────
    if args.storage == "postgres":
        print("  PostgreSQL setup\n")
        host = args.host or input("  Host [localhost]: ").strip() or "localhost"
        port = args.port or input("  Port [5432]: ").strip() or "5432"
        database = args.database or input("  Database [alma]: ").strip() or "alma"
        user = args.user or input("  User [postgres]: ").strip() or "postgres"
        project_id = args.project or Path.cwd().name or "my-project"

        config_path.write_text(
            POSTGRES_CONFIG.format(
                project_id=project_id,
                host=host,
                port=port,
                database=database,
                user=user,
            )
        )
        print(f"\n  {_green('✓')}  Config written: {config_path}")
        print(f"  {_yellow('!')}  Set password:   export ALMA_DB_PASSWORD='your-password'")
        print(f"  {_yellow('!')}  Enable pgvector: alma pg-setup --host {host} --port {port} --db {database} --user {user}")
        print(f"\n  Then run: alma test --config {config_path}\n")
        return 0

    # ── interactive ───────────────────────────────────────────────────────────
    print("  Choose storage backend:")
    print("  1) SQLite   — zero infrastructure, works immediately (recommended)")
    print("  2) PostgreSQL — production, team sharing")
    print("  3) Qdrant   — dedicated vector search")
    print()
    choice = input("  Choice [1]: ").strip() or "1"

    if choice == "1":
        args.quickstart = True
        args.storage = "sqlite"
        return cmd_init(args)
    elif choice == "2":
        args.storage = "postgres"
        return cmd_init(args)
    else:
        print(_yellow("  Only SQLite and PostgreSQL supported in guided setup. See GUIDE.md for others."))
        args.quickstart = True
        args.storage = "sqlite"
        return cmd_init(args)


# ─── doctor ───────────────────────────────────────────────────────────────────

def cmd_doctor(args: argparse.Namespace) -> int:
    print(_bold("\nALMA Doctor\n"))
    all_ok = True

    # Python version
    v = sys.version_info
    ok = v >= (3, 10)
    all_ok &= _check(f"Python {v.major}.{v.minor}.{v.micro}", ok,
                     "Need Python 3.10+: https://python.org/downloads")

    # alma-memory installed
    try:
        import alma
        _check(f"alma-memory installed (v{alma.__version__})", True)
    except (ImportError, AttributeError):
        all_ok &= _check("alma-memory installed", False, "pip install alma-memory")

    # sentence-transformers (local embeddings)
    try:
        import sentence_transformers
        _check(f"sentence-transformers {sentence_transformers.__version__} (local embeddings)", True)
    except ImportError:
        _check("sentence-transformers (local embeddings)", False,
               "pip install 'alma-memory[local]'")
        # Not fatal — user may use azure or postgres embeddings

    # faiss
    try:
        import faiss
        _check("faiss (fast vector search)", True)
    except ImportError:
        _check("faiss (fast vector search)", False,
               "pip install 'alma-memory[local]'  — uses numpy fallback until then")

    # psycopg (postgres)
    try:
        import psycopg
        _check(f"psycopg {psycopg.__version__} (PostgreSQL support)", True)
    except ImportError:
        _check("psycopg (PostgreSQL support)", False,
               "pip install 'alma-memory[postgres]'  — only needed if using PostgreSQL")

    # config file
    config_path = Path(args.config).expanduser()
    if config_path.exists():
        _check(f"Config found: {config_path}", True)
    else:
        _check(f"Config found ({config_path})", False,
               "Run: alma init --quickstart")
        all_ok = False

    # live test
    if config_path.exists():
        try:
            from alma import ALMA
            alma_inst = ALMA.from_config(str(config_path))
            alma_inst.learn(agent="doctor", task="health check", outcome="success",
                           strategy_used="alma doctor")
            result = alma_inst.retrieve(task="health", agent="doctor", top_k=1)
            _check("Live store+retrieve test", True)
        except Exception as e:
            all_ok &= _check("Live store+retrieve test", False, str(e)[:120])

    print()
    if all_ok:
        print(_green("  All checks passed. ALMA is ready.\n"))
        return 0
    else:
        print(_yellow("  Fix the issues above, then re-run: alma doctor\n"))
        return 1


# ─── test ─────────────────────────────────────────────────────────────────────

def cmd_test(args: argparse.Namespace) -> int:
    print(_bold("\nALMA Test\n"))

    config_path = Path(args.config).expanduser()
    if not config_path.exists():
        print(_red(f"  No config at {config_path}"))
        print(f"  Run: alma init --quickstart\n")
        return 1

    try:
        from alma import ALMA
        print(f"  Loading config: {config_path}")
        alma = ALMA.from_config(str(config_path))

        print("  Storing test memory...")
        alma.learn(
            agent="alma-test",
            task="x402 payment handler deployment",
            outcome="success",
            strategy_used="Blue-green deployment — zero downtime, instant rollback",
        )
        alma.learn(
            agent="alma-test",
            task="payment processor timeout",
            outcome="failure",
            strategy_used="Synchronous retry — caused cascade",
        )

        print("  Retrieving memories...")
        result = alma.retrieve(task="deploy payment service", agent="alma-test", top_k=3)

        print()
        print(f"  {_green('✓')}  Outcomes stored and retrieved: {len(result.outcomes)}")
        print(f"  {_green('✓')}  Storage backend: working")
        print(f"  {_green('✓')}  Embeddings:      working")
        print()
        print(_green("  ALMA is working correctly.\n"))
        return 0

    except ImportError as e:
        if "sentence-transformers" in str(e):
            print(_red("  Missing local embeddings:"))
            print(f"    pip install 'alma-memory[local]'\n")
        else:
            print(_red(f"  Import error: {e}\n"))
        return 1
    except Exception as e:
        print(_red(f"  Test failed: {e}\n"))
        return 1


# ─── pg-setup ─────────────────────────────────────────────────────────────────

PG_SETUP_SQL = """\
-- ALMA PostgreSQL Setup
-- Run this once against your database to prepare it for ALMA.
-- Requires PostgreSQL 14+ with pgvector extension available.

-- Step 1: Enable pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Step 2: Core memory tables (ALMA auto-creates these, but run manually for production)
CREATE TABLE IF NOT EXISTS alma_heuristics (
    id TEXT PRIMARY KEY,
    agent TEXT NOT NULL,
    project_id TEXT NOT NULL,
    condition TEXT NOT NULL,
    strategy TEXT NOT NULL,
    confidence REAL DEFAULT 0.0,
    occurrence_count INTEGER DEFAULT 0,
    success_count INTEGER DEFAULT 0,
    last_validated TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB,
    embedding VECTOR(384)
);

CREATE TABLE IF NOT EXISTS alma_outcomes (
    id TEXT PRIMARY KEY,
    agent TEXT NOT NULL,
    project_id TEXT NOT NULL,
    task_type TEXT,
    task_description TEXT NOT NULL,
    success BOOLEAN DEFAULT FALSE,
    strategy_used TEXT,
    duration_ms INTEGER,
    error_message TEXT,
    user_feedback TEXT,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB,
    embedding VECTOR(384)
);

CREATE TABLE IF NOT EXISTS alma_domain_knowledge (
    id TEXT PRIMARY KEY,
    agent TEXT NOT NULL,
    project_id TEXT NOT NULL,
    domain TEXT,
    fact TEXT NOT NULL,
    source TEXT,
    confidence REAL DEFAULT 1.0,
    last_verified TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB,
    embedding VECTOR(384)
);

CREATE TABLE IF NOT EXISTS alma_anti_patterns (
    id TEXT PRIMARY KEY,
    agent TEXT NOT NULL,
    project_id TEXT NOT NULL,
    pattern TEXT NOT NULL,
    why_bad TEXT,
    better_alternative TEXT,
    occurrence_count INTEGER DEFAULT 1,
    last_seen TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB,
    embedding VECTOR(384)
);

CREATE TABLE IF NOT EXISTS alma_preferences (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    category TEXT,
    preference TEXT NOT NULL,
    source TEXT,
    confidence REAL DEFAULT 1.0,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB
);

-- Step 3: Indexes for performance
CREATE INDEX IF NOT EXISTS idx_heuristics_project_agent ON alma_heuristics(project_id, agent);
CREATE INDEX IF NOT EXISTS idx_heuristics_confidence ON alma_heuristics(project_id, confidence DESC);
CREATE INDEX IF NOT EXISTS idx_outcomes_project_agent ON alma_outcomes(project_id, agent);
CREATE INDEX IF NOT EXISTS idx_outcomes_timestamp ON alma_outcomes(project_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_domain_project_agent ON alma_domain_knowledge(project_id, agent);
CREATE INDEX IF NOT EXISTS idx_antipatterns_project_agent ON alma_anti_patterns(project_id, agent);
CREATE INDEX IF NOT EXISTS idx_preferences_user ON alma_preferences(user_id);

-- Done. Your database is ready for ALMA.
-- Next: configure .alma/config.yaml and run: alma test
"""


def cmd_pg_setup(args: argparse.Namespace) -> int:
    print(_bold("\nALMA PostgreSQL Setup\n"))

    if args.print_sql:
        print(PG_SETUP_SQL)
        return 0

    if args.run:
        host = args.host or "localhost"
        port = args.port or "5432"
        database = args.database or "postgres"
        user = args.user or "postgres"

        print(f"  Connecting to {user}@{host}:{port}/{database} ...")

        try:
            result = subprocess.run(
                ["psql", f"--host={host}", f"--port={port}",
                 f"--dbname={database}", f"--username={user}",
                 "--command", PG_SETUP_SQL],
                capture_output=True, text=True, check=True,
            )
            print(_green("  ✓  PostgreSQL setup complete."))
            print(f"  ✓  pgvector enabled, all tables created.\n")
            return 0
        except subprocess.CalledProcessError as e:
            print(_red(f"  psql error: {e.stderr[:200]}"))
            print(_yellow("  Try: alma pg-setup --print-sql | psql -h HOST -U USER -d DB\n"))
            return 1
        except FileNotFoundError:
            print(_red("  psql not found in PATH."))
            print(_yellow("  Use: alma pg-setup --print-sql | psql -h HOST -U USER -d DB"))
            print(_yellow("  Or paste the SQL into your database console.\n"))
            return 1

    # Default: print instructions
    print("  Two options:\n")
    print(f"  {_bold('Option A — run automatically (needs psql in PATH):')}")
    print("    alma pg-setup --run --host HOST --port 5432 --db DATABASE --user USER\n")
    print(f"  {_bold('Option B — print SQL and run yourself:')}")
    print("    alma pg-setup --print-sql > setup.sql")
    print("    # Then paste setup.sql into Supabase SQL editor, psql, or any PG console\n")
    return 0


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="alma",
        description="ALMA — Agent Learning Memory Architecture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  init         Set up ALMA in the current project
  doctor       Check your ALMA installation
  test         Run a live store+retrieve test
  pg-setup     Set up PostgreSQL for ALMA

Examples:
  alma init --quickstart          # 10-second SQLite setup
  alma init --storage postgres    # Guided PostgreSQL setup
  alma doctor                     # Check everything is working
  alma test                       # Run a live test
  alma pg-setup --run --host localhost --db mydb --user postgres
  alma pg-setup --print-sql       # Print setup SQL for Supabase / managed DB
        """,
    )

    sub = parser.add_subparsers(dest="command")

    # init
    p_init = sub.add_parser("init", help="Set up ALMA in the current project")
    p_init.add_argument("--quickstart", action="store_true",
                        help="Zero-config SQLite setup, no questions asked")
    p_init.add_argument("--storage", choices=["sqlite", "postgres"],
                        help="Storage backend to configure")
    p_init.add_argument("--dir", default=".alma",
                        help="Directory for config and database (default: .alma)")
    p_init.add_argument("--project", default="",
                        help="Project ID (default: current directory name)")
    p_init.add_argument("--force", action="store_true",
                        help="Overwrite existing config")
    p_init.add_argument("--host", default="", help="PostgreSQL host")
    p_init.add_argument("--port", default="", help="PostgreSQL port")
    p_init.add_argument("--database", default="", help="PostgreSQL database name")
    p_init.add_argument("--user", default="", help="PostgreSQL user")

    # doctor
    p_doc = sub.add_parser("doctor", help="Check your ALMA installation")
    p_doc.add_argument("--config", default=".alma/config.yaml",
                       help="Path to config file (default: .alma/config.yaml)")

    # test
    p_test = sub.add_parser("test", help="Run a live store+retrieve test")
    p_test.add_argument("--config", default=".alma/config.yaml",
                        help="Path to config file (default: .alma/config.yaml)")

    # pg-setup
    p_pg = sub.add_parser("pg-setup", help="Set up PostgreSQL for ALMA")
    p_pg.add_argument("--print-sql", action="store_true",
                      help="Print setup SQL to stdout (for Supabase, etc.)")
    p_pg.add_argument("--run", action="store_true",
                      help="Run setup SQL directly via psql")
    p_pg.add_argument("--host", default="", help="PostgreSQL host")
    p_pg.add_argument("--port", default="", help="PostgreSQL port")
    p_pg.add_argument("--database", default="", help="PostgreSQL database name")
    p_pg.add_argument("--user", default="", help="PostgreSQL user")

    args = parser.parse_args()

    if args.command == "init":
        sys.exit(cmd_init(args))
    elif args.command == "doctor":
        sys.exit(cmd_doctor(args))
    elif args.command == "test":
        sys.exit(cmd_test(args))
    elif args.command == "pg-setup":
        sys.exit(cmd_pg_setup(args))
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
