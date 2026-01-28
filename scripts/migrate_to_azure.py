#!/usr/bin/env python3
"""
Migration script from local SQLite storage to Azure Cosmos DB.

This script transfers all ALMA memory data from a local SQLite database
to Azure Cosmos DB for cloud deployment.

Usage:
    python scripts/migrate_to_azure.py --sqlite-path ./data/alma.db \
        --cosmos-endpoint https://your-account.documents.azure.com:443/ \
        --cosmos-key <your-key>

    # Dry run (no writes):
    python scripts/migrate_to_azure.py --sqlite-path ./data/alma.db \
        --cosmos-endpoint https://your-account.documents.azure.com:443/ \
        --cosmos-key <your-key> --dry-run

    # Using Azure Key Vault:
    python scripts/migrate_to_azure.py --sqlite-path ./data/alma.db \
        --config ./config/azure.yaml
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from alma.config.loader import ConfigLoader
from alma.storage.azure_cosmos import AzureCosmosStorage
from alma.storage.sqlite_local import SQLiteStorage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class MigrationStats:
    """Track migration statistics."""

    def __init__(self):
        self.heuristics_migrated = 0
        self.outcomes_migrated = 0
        self.preferences_migrated = 0
        self.knowledge_migrated = 0
        self.antipatterns_migrated = 0
        self.errors = 0

    def total(self) -> int:
        return (
            self.heuristics_migrated
            + self.outcomes_migrated
            + self.preferences_migrated
            + self.knowledge_migrated
            + self.antipatterns_migrated
        )

    def summary(self) -> str:
        return (
            f"\nMigration Summary:\n"
            f"  Heuristics:     {self.heuristics_migrated}\n"
            f"  Outcomes:       {self.outcomes_migrated}\n"
            f"  Preferences:    {self.preferences_migrated}\n"
            f"  Knowledge:      {self.knowledge_migrated}\n"
            f"  Anti-patterns:  {self.antipatterns_migrated}\n"
            f"  Total:          {self.total()}\n"
            f"  Errors:         {self.errors}"
        )


def migrate_heuristics(
    source: SQLiteStorage,
    target: AzureCosmosStorage,
    project_ids: list,
    dry_run: bool,
    stats: MigrationStats,
) -> None:
    """Migrate all heuristics."""
    logger.info("Migrating heuristics...")

    for project_id in project_ids:
        heuristics = source.get_heuristics(project_id=project_id, top_k=10000)
        for h in heuristics:
            try:
                if not dry_run:
                    target.save_heuristic(h)
                stats.heuristics_migrated += 1
                if stats.heuristics_migrated % 100 == 0:
                    logger.info(f"  Migrated {stats.heuristics_migrated} heuristics...")
            except Exception as e:
                logger.error(f"  Error migrating heuristic {h.id}: {e}")
                stats.errors += 1


def migrate_outcomes(
    source: SQLiteStorage,
    target: AzureCosmosStorage,
    project_ids: list,
    dry_run: bool,
    stats: MigrationStats,
) -> None:
    """Migrate all outcomes."""
    logger.info("Migrating outcomes...")

    for project_id in project_ids:
        outcomes = source.get_outcomes(project_id=project_id, limit=10000)
        for o in outcomes:
            try:
                if not dry_run:
                    target.save_outcome(o)
                stats.outcomes_migrated += 1
                if stats.outcomes_migrated % 100 == 0:
                    logger.info(f"  Migrated {stats.outcomes_migrated} outcomes...")
            except Exception as e:
                logger.error(f"  Error migrating outcome {o.id}: {e}")
                stats.errors += 1


def migrate_preferences(
    source: SQLiteStorage,
    target: AzureCosmosStorage,
    user_ids: list,
    dry_run: bool,
    stats: MigrationStats,
) -> None:
    """Migrate all user preferences."""
    logger.info("Migrating user preferences...")

    for user_id in user_ids:
        prefs = source.get_user_preferences(user_id=user_id)
        for p in prefs:
            try:
                if not dry_run:
                    target.save_user_preference(p)
                stats.preferences_migrated += 1
            except Exception as e:
                logger.error(f"  Error migrating preference {p.id}: {e}")
                stats.errors += 1

    logger.info(f"  Migrated {stats.preferences_migrated} preferences")


def migrate_knowledge(
    source: SQLiteStorage,
    target: AzureCosmosStorage,
    project_ids: list,
    dry_run: bool,
    stats: MigrationStats,
) -> None:
    """Migrate all domain knowledge."""
    logger.info("Migrating domain knowledge...")

    for project_id in project_ids:
        knowledge = source.get_domain_knowledge(project_id=project_id, top_k=10000)
        for k in knowledge:
            try:
                if not dry_run:
                    target.save_domain_knowledge(k)
                stats.knowledge_migrated += 1
                if stats.knowledge_migrated % 100 == 0:
                    logger.info(f"  Migrated {stats.knowledge_migrated} knowledge items...")
            except Exception as e:
                logger.error(f"  Error migrating knowledge {k.id}: {e}")
                stats.errors += 1


def migrate_antipatterns(
    source: SQLiteStorage,
    target: AzureCosmosStorage,
    project_ids: list,
    dry_run: bool,
    stats: MigrationStats,
) -> None:
    """Migrate all anti-patterns."""
    logger.info("Migrating anti-patterns...")

    for project_id in project_ids:
        antipatterns = source.get_anti_patterns(project_id=project_id, top_k=10000)
        for ap in antipatterns:
            try:
                if not dry_run:
                    target.save_anti_pattern(ap)
                stats.antipatterns_migrated += 1
            except Exception as e:
                logger.error(f"  Error migrating anti-pattern {ap.id}: {e}")
                stats.errors += 1

    logger.info(f"  Migrated {stats.antipatterns_migrated} anti-patterns")


def get_all_project_ids(source: SQLiteStorage) -> list:
    """Get all unique project IDs from the source database."""
    project_ids = set()

    # Query each table for unique project_ids
    conn = source._get_connection()
    try:
        for table in ["heuristics", "outcomes", "domain_knowledge", "anti_patterns"]:
            cursor = conn.execute(f"SELECT DISTINCT project_id FROM {table}")
            for row in cursor:
                if row[0]:
                    project_ids.add(row[0])
    finally:
        conn.close()

    return list(project_ids)


def get_all_user_ids(source: SQLiteStorage) -> list:
    """Get all unique user IDs from the source database."""
    user_ids = set()

    conn = source._get_connection()
    try:
        cursor = conn.execute("SELECT DISTINCT user_id FROM user_preferences")
        for row in cursor:
            if row[0]:
                user_ids.add(row[0])
    finally:
        conn.close()

    return list(user_ids)


def run_migration(
    sqlite_path: str,
    cosmos_endpoint: str,
    cosmos_key: str,
    database_name: str = "alma-memory",
    dry_run: bool = False,
    embedding_dim: int = 384,
) -> MigrationStats:
    """Run the full migration."""
    stats = MigrationStats()

    # Initialize source
    logger.info(f"Connecting to SQLite: {sqlite_path}")
    source = SQLiteStorage(db_path=sqlite_path, embedding_dim=embedding_dim)

    # Get all project and user IDs
    project_ids = get_all_project_ids(source)
    user_ids = get_all_user_ids(source)

    logger.info(f"Found {len(project_ids)} projects and {len(user_ids)} users to migrate")

    if not project_ids and not user_ids:
        logger.warning("No data found to migrate!")
        return stats

    # Initialize target
    if dry_run:
        logger.info("DRY RUN MODE - No data will be written to Azure")
        # Create a mock target for dry run
        target = None
    else:
        logger.info(f"Connecting to Cosmos DB: {cosmos_endpoint}")
        target = AzureCosmosStorage(
            endpoint=cosmos_endpoint,
            key=cosmos_key,
            database_name=database_name,
            embedding_dim=embedding_dim,
            create_if_not_exists=True,
        )

    # Run migrations
    if target or dry_run:
        migrate_heuristics(source, target, project_ids, dry_run, stats)
        migrate_outcomes(source, target, project_ids, dry_run, stats)
        migrate_preferences(source, target, user_ids, dry_run, stats)
        migrate_knowledge(source, target, project_ids, dry_run, stats)
        migrate_antipatterns(source, target, project_ids, dry_run, stats)

    return stats


def load_config_credentials(config_path: str) -> tuple:
    """Load Cosmos DB credentials from config file."""
    loader = ConfigLoader()
    config = loader.load(config_path)

    storage_config = config.get("storage", {}).get("azure", {})
    endpoint = storage_config.get("endpoint")
    key = storage_config.get("key")
    database = storage_config.get("database", "alma-memory")

    if not endpoint or not key:
        raise ValueError(
            "Config file must contain storage.azure.endpoint and storage.azure.key"
        )

    return endpoint, key, database


def main():
    parser = argparse.ArgumentParser(
        description="Migrate ALMA data from SQLite to Azure Cosmos DB"
    )

    # Source options
    parser.add_argument(
        "--sqlite-path",
        type=str,
        required=True,
        help="Path to SQLite database file",
    )

    # Target options (direct)
    parser.add_argument(
        "--cosmos-endpoint",
        type=str,
        help="Azure Cosmos DB endpoint URL",
    )
    parser.add_argument(
        "--cosmos-key",
        type=str,
        help="Azure Cosmos DB key",
    )
    parser.add_argument(
        "--database-name",
        type=str,
        default="alma-memory",
        help="Cosmos DB database name (default: alma-memory)",
    )

    # Target options (config file)
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config file with Azure credentials (uses Key Vault if configured)",
    )

    # Migration options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count items without writing to Azure",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=384,
        help="Embedding dimension (default: 384 for all-MiniLM-L6-v2)",
    )

    args = parser.parse_args()

    # Validate arguments
    if not Path(args.sqlite_path).exists():
        logger.error(f"SQLite file not found: {args.sqlite_path}")
        sys.exit(1)

    # Get Cosmos credentials
    cosmos_endpoint: Optional[str] = None
    cosmos_key: Optional[str] = None
    database_name = args.database_name

    if args.config:
        try:
            cosmos_endpoint, cosmos_key, database_name = load_config_credentials(
                args.config
            )
            logger.info("Loaded credentials from config file")
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            sys.exit(1)
    elif args.cosmos_endpoint and args.cosmos_key:
        cosmos_endpoint = args.cosmos_endpoint
        cosmos_key = args.cosmos_key
    elif not args.dry_run:
        logger.error(
            "Must provide either --cosmos-endpoint and --cosmos-key, "
            "or --config, or use --dry-run"
        )
        sys.exit(1)

    # Run migration
    try:
        stats = run_migration(
            sqlite_path=args.sqlite_path,
            cosmos_endpoint=cosmos_endpoint or "",
            cosmos_key=cosmos_key or "",
            database_name=database_name,
            dry_run=args.dry_run,
            embedding_dim=args.embedding_dim,
        )

        logger.info(stats.summary())

        if stats.errors > 0:
            logger.warning(f"Migration completed with {stats.errors} errors")
            sys.exit(1)
        else:
            logger.info("Migration completed successfully!")

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
