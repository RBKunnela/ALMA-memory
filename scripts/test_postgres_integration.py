#!/usr/bin/env python3
"""
Test PostgreSQL backend against Azure PostgreSQL.

This script:
1. Connects to Azure PostgreSQL
2. Creates ALMA tables (with pgvector if available)
3. Tests basic CRUD operations
4. Reports results
"""

import os
import sys
from datetime import datetime, timezone

# Add alma to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alma.storage.postgresql import (
    NUMPY_AVAILABLE,
    PSYCOPG_AVAILABLE,
    PostgreSQLStorage,
)
from alma.types import AntiPattern, DomainKnowledge, Heuristic, Outcome, UserPreference

# Azure PostgreSQL credentials
POSTGRES_HOST = "psql-agentictestari-dev.postgres.database.azure.com"
POSTGRES_PORT = 5432
POSTGRES_DATABASE = "agentictestari"
POSTGRES_USER = "pgadmin"
POSTGRES_PASSWORD = "Armour2026Secure!"
POSTGRES_SSL_MODE = "require"


def test_connection():
    """Test basic connection to Azure PostgreSQL."""
    print("\n" + "=" * 60)
    print("ALMA PostgreSQL Backend Integration Test")
    print("=" * 60)

    print("\n📦 Dependencies:")
    print(f"   psycopg available: {PSYCOPG_AVAILABLE}")
    print(f"   numpy available: {NUMPY_AVAILABLE}")

    if not PSYCOPG_AVAILABLE:
        print("\n❌ psycopg not installed. Run: pip install 'psycopg[binary,pool]'")
        return False

    print("\n🔌 Connecting to Azure PostgreSQL...")
    print(f"   Host: {POSTGRES_HOST}")
    print(f"   Database: {POSTGRES_DATABASE}")
    print(f"   User: {POSTGRES_USER}")
    print(f"   SSL: {POSTGRES_SSL_MODE}")

    try:
        storage = PostgreSQLStorage(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            database=POSTGRES_DATABASE,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            embedding_dim=384,
            pool_size=5,
            ssl_mode=POSTGRES_SSL_MODE,
        )
        print("   ✅ Connection successful!")
        print(f"   pgvector available: {storage._pgvector_available}")
        return storage
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
        return None


def test_crud_operations(storage: PostgreSQLStorage):
    """Test CRUD operations."""
    print("\n" + "-" * 60)
    print("Testing CRUD Operations")
    print("-" * 60)

    project_id = "test-alma-integration"

    # Test 1: Save Heuristic
    print("\n📝 Test 1: Save Heuristic")
    try:
        heuristic = Heuristic(
            id="test-h-001",
            agent="Helena",
            project_id=project_id,
            condition="form with multiple required fields",
            strategy="validate each field individually before submitting",
            confidence=0.85,
            occurrence_count=10,
            success_count=8,
            last_validated=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
            metadata={"tags": ["forms", "validation"]},
        )
        result = storage.save_heuristic(heuristic)
        print(f"   ✅ Saved heuristic: {result}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False

    # Test 2: Save Outcome
    print("\n📝 Test 2: Save Outcome")
    try:
        outcome = Outcome(
            id="test-o-001",
            agent="Victor",
            project_id=project_id,
            task_type="api_testing",
            task_description="Test login endpoint with valid credentials",
            success=True,
            strategy_used="happy_path_first",
            duration_ms=150,
            error_message=None,
            user_feedback=None,
            timestamp=datetime.now(timezone.utc),
            metadata={"endpoint": "/api/auth/login"},
        )
        result = storage.save_outcome(outcome)
        print(f"   ✅ Saved outcome: {result}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False

    # Test 3: Save Domain Knowledge
    print("\n📝 Test 3: Save Domain Knowledge")
    try:
        knowledge = DomainKnowledge(
            id="test-dk-001",
            agent="Helena",
            project_id=project_id,
            domain="authentication",
            fact="Login session expires after 30 minutes of inactivity",
            source="code_analysis",
            confidence=0.95,
            last_verified=datetime.now(timezone.utc),
            metadata={"source_file": "auth.config.ts"},
        )
        result = storage.save_domain_knowledge(knowledge)
        print(f"   ✅ Saved domain knowledge: {result}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False

    # Test 4: Save Anti-Pattern
    print("\n📝 Test 4: Save Anti-Pattern")
    try:
        anti_pattern = AntiPattern(
            id="test-ap-001",
            agent="Victor",
            project_id=project_id,
            pattern="Using fixed sleep() for async API waits",
            why_bad="Causes flaky tests and slow execution",
            better_alternative="Use explicit waits with timeout conditions",
            occurrence_count=5,
            last_seen=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
            metadata={"severity": "high"},
        )
        result = storage.save_anti_pattern(anti_pattern)
        print(f"   ✅ Saved anti-pattern: {result}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False

    # Test 5: Save User Preference
    print("\n📝 Test 5: Save User Preference")
    try:
        preference = UserPreference(
            id="test-up-001",
            user_id="test-user",
            category="communication",
            preference="No emojis in test reports",
            source="explicit_instruction",
            confidence=1.0,
            timestamp=datetime.now(timezone.utc),
            metadata={"context": "QA documentation"},
        )
        result = storage.save_user_preference(preference)
        print(f"   ✅ Saved user preference: {result}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False

    # Test 6: Get Heuristics
    print("\n📖 Test 6: Get Heuristics")
    try:
        heuristics = storage.get_heuristics(project_id=project_id, agent="Helena", top_k=5)
        print(f"   ✅ Retrieved {len(heuristics)} heuristic(s)")
        for h in heuristics:
            print(f"      - {h.id}: {h.condition[:50]}...")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False

    # Test 7: Get Outcomes
    print("\n📖 Test 7: Get Outcomes")
    try:
        outcomes = storage.get_outcomes(project_id=project_id, agent="Victor", top_k=5)
        print(f"   ✅ Retrieved {len(outcomes)} outcome(s)")
        for o in outcomes:
            print(f"      - {o.id}: {o.task_description[:50]}... (success={o.success})")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False

    # Test 8: Get Stats
    print("\n📊 Test 8: Get Stats")
    try:
        stats = storage.get_stats(project_id=project_id)
        print("   ✅ Stats retrieved:")
        print(f"      - Storage type: {stats['storage_type']}")
        print(f"      - pgvector: {stats['pgvector_available']}")
        print(f"      - Heuristics: {stats['heuristics_count']}")
        print(f"      - Outcomes: {stats['outcomes_count']}")
        print(f"      - Domain knowledge: {stats['domain_knowledge_count']}")
        print(f"      - Anti-patterns: {stats['anti_patterns_count']}")
        print(f"      - Preferences: {stats['preferences_count']}")
        print(f"      - Total: {stats['total_count']}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False

    # Test 9: Update Heuristic
    print("\n✏️ Test 9: Update Heuristic Confidence")
    try:
        result = storage.update_heuristic_confidence("test-h-001", 0.92)
        print(f"   ✅ Updated confidence: {result}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False

    # Test 10: Increment Occurrence
    print("\n➕ Test 10: Increment Heuristic Occurrence")
    try:
        result = storage.increment_heuristic_occurrence("test-h-001", success=True)
        print(f"   ✅ Incremented occurrence: {result}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False

    return True


def cleanup_test_data(storage: PostgreSQLStorage):
    """Clean up test data."""
    print("\n🧹 Cleaning up test data...")
    try:
        storage.delete_heuristic("test-h-001")
        storage.delete_outcome("test-o-001")
        storage.delete_domain_knowledge("test-dk-001")
        storage.delete_anti_pattern("test-ap-001")

        # Delete preference manually (not in base class)
        with storage._get_connection() as conn:
            conn.execute(f"DELETE FROM {storage.schema}.alma_preferences WHERE id = %s", ("test-up-001",))
            conn.commit()

        print("   ✅ Test data cleaned up")
    except Exception as e:
        print(f"   ⚠️ Cleanup warning: {e}")


def main():
    """Run all tests."""
    storage = test_connection()

    if not storage:
        print("\n❌ FAILED: Could not connect to database")
        return 1

    success = test_crud_operations(storage)

    if success:
        cleanup_test_data(storage)
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        print("\nALMA PostgreSQL backend is ready for production!")
        print(f"pgvector support: {'YES' if storage._pgvector_available else 'NO (using fallback)'}")
        return 0
    else:
        print("\n" + "=" * 60)
        print("❌ SOME TESTS FAILED")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
