"""
Tests for the Session Initializer module.
"""

import subprocess
import tempfile
from pathlib import Path

import pytest

from alma.initializer import (
    CodebaseOrientation,
    InitializationResult,
    RulesOfEngagement,
    SessionInitializer,
)


class TestCodebaseOrientation:
    """Tests for CodebaseOrientation dataclass."""

    def test_basic_creation(self):
        """Test basic orientation creation."""
        orientation = CodebaseOrientation(
            current_branch="main",
            has_uncommitted_changes=False,
            recent_commits=["abc123 Initial commit"],
            root_path="/path/to/project",
            key_directories=["src", "tests"],
            config_files=["pyproject.toml"],
        )

        assert orientation.current_branch == "main"
        assert orientation.has_uncommitted_changes is False
        assert len(orientation.recent_commits) == 1

    def test_to_prompt(self):
        """Test prompt formatting."""
        orientation = CodebaseOrientation(
            current_branch="feature/test",
            has_uncommitted_changes=True,
            recent_commits=["abc123 Add feature", "def456 Fix bug"],
            root_path="/project",
            key_directories=["src"],
            config_files=["package.json"],
            summary="Node.js project",
        )

        prompt = orientation.to_prompt()
        assert "feature/test" in prompt
        assert "Yes" in prompt  # Uncommitted changes
        assert "Add feature" in prompt


class TestRulesOfEngagement:
    """Tests for RulesOfEngagement dataclass."""

    def test_basic_creation(self):
        """Test basic rules creation."""
        rules = RulesOfEngagement(
            scope_rules=["Test UI components"],
            constraints=["Do not modify backend"],
            quality_gates=["All tests pass"],
        )

        assert len(rules.scope_rules) == 1
        assert len(rules.constraints) == 1
        assert len(rules.quality_gates) == 1

    def test_to_prompt(self):
        """Test prompt formatting."""
        rules = RulesOfEngagement(
            scope_rules=["Test login flow"],
            constraints=["No API changes"],
            quality_gates=["Coverage > 80%"],
        )

        prompt = rules.to_prompt()
        assert "You CAN:" in prompt
        assert "You CANNOT:" in prompt
        assert "Before marking DONE" in prompt

    def test_empty_rules(self):
        """Test empty rules to_prompt."""
        rules = RulesOfEngagement()
        prompt = rules.to_prompt()
        assert prompt == ""


class TestInitializationResult:
    """Tests for InitializationResult dataclass."""

    def test_create(self):
        """Test creating initialization result."""
        result = InitializationResult.create(
            project_id="proj-1",
            agent="Helena",
            original_prompt="Test the login form",
        )

        assert result.id is not None
        assert result.session_id is not None
        assert result.project_id == "proj-1"
        assert result.agent == "Helena"
        assert result.original_prompt == "Test the login form"
        assert result.goal == "Test the login form"  # Default to original

    def test_create_with_goal(self):
        """Test creating with custom goal."""
        result = InitializationResult.create(
            project_id="proj-1",
            agent="Victor",
            original_prompt="Fix auth",
            goal="Fix authentication flow in login endpoint",
        )

        assert result.original_prompt == "Fix auth"
        assert result.goal == "Fix authentication flow in login endpoint"

    def test_to_prompt(self):
        """Test prompt generation."""
        result = InitializationResult.create(
            project_id="test-project",
            agent="Helena",
            original_prompt="Test login",
            goal="Verify login form validation",
        )

        result.orientation = CodebaseOrientation(
            current_branch="main",
            has_uncommitted_changes=False,
            recent_commits=["abc123 Add login"],
            root_path="/project",
            key_directories=["src"],
            config_files=[],
        )

        result.rules = RulesOfEngagement(
            scope_rules=["Test UI"],
            constraints=["No backend"],
            quality_gates=["Tests pass"],
        )

        prompt = result.to_prompt()

        assert "Session Initialization for Helena" in prompt
        assert "test-project" in prompt
        assert "Verify login form validation" in prompt
        assert "Codebase Orientation" in prompt
        assert "Rules of Engagement" in prompt

    def test_to_dict(self):
        """Test serialization."""
        result = InitializationResult.create(
            project_id="proj",
            agent="Agent",
            original_prompt="Do task",
        )

        data = result.to_dict()

        assert data["project_id"] == "proj"
        assert data["agent"] == "Agent"
        assert data["original_prompt"] == "Do task"
        assert "initialized_at" in data


class TestSessionInitializer:
    """Tests for SessionInitializer."""

    @pytest.fixture
    def initializer(self):
        """Create a basic initializer."""
        return SessionInitializer()

    def test_expand_prompt_bullet_points(self, initializer):
        """Test expanding bullet point prompts."""
        prompt = """
        Test the following:
        - Login form validation
        - Password reset flow
        - Session timeout
        """

        items = initializer.expand_prompt(prompt)

        assert len(items) == 3
        assert items[0].title == "Login form validation"
        assert items[1].title == "Password reset flow"
        assert items[2].title == "Session timeout"

    def test_expand_prompt_numbered(self, initializer):
        """Test expanding numbered prompts."""
        prompt = """
        1. Create user model
        2. Add authentication endpoint
        3. Write unit tests
        """

        items = initializer.expand_prompt(prompt)

        assert len(items) == 3
        assert items[0].title == "Create user model"

    def test_expand_prompt_simple(self, initializer):
        """Test expanding simple prompt without structure."""
        prompt = "Fix the login bug"

        items = initializer.expand_prompt(prompt)

        assert len(items) == 1
        assert items[0].title == "Fix the login bug"

    def test_expand_prompt_long_title(self, initializer):
        """Test truncation of long prompts."""
        prompt = "A" * 200  # Very long prompt

        items = initializer.expand_prompt(prompt)

        assert len(items) == 1
        assert len(items[0].title) <= 103  # 100 + "..."

    def test_orient_to_codebase_non_git(self, initializer):
        """Test orientation for non-git directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orientation = initializer.orient_to_codebase(tmpdir)

            assert orientation.current_branch == "unknown"
            # Summary still generated, just no git info
            assert "unknown" in orientation.summary

    def test_orient_to_codebase_git_repo(self, initializer):
        """Test orientation for git repository."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Initialize git repo
            subprocess.run(["git", "init"], cwd=tmpdir, capture_output=True)
            subprocess.run(
                ["git", "config", "user.email", "test@test.com"],
                cwd=tmpdir,
                capture_output=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Test"], cwd=tmpdir, capture_output=True
            )

            # Create a file and commit
            (Path(tmpdir) / "README.md").write_text("# Test")
            subprocess.run(["git", "add", "."], cwd=tmpdir, capture_output=True)
            subprocess.run(
                ["git", "commit", "-m", "Initial commit"],
                cwd=tmpdir,
                capture_output=True,
            )

            orientation = initializer.orient_to_codebase(tmpdir)

            # Should detect git info
            assert orientation.current_branch in ["main", "master"]
            assert orientation.has_uncommitted_changes is False
            assert len(orientation.recent_commits) >= 1
            assert "Initial commit" in orientation.recent_commits[0]

    def test_orient_finds_key_directories(self, initializer):
        """Test detection of key directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create key directories
            (Path(tmpdir) / "src").mkdir()
            (Path(tmpdir) / "tests").mkdir()

            orientation = initializer.orient_to_codebase(tmpdir)

            assert "src" in orientation.key_directories
            assert "tests" in orientation.key_directories

    def test_orient_finds_config_files(self, initializer):
        """Test detection of config files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "pyproject.toml").write_text("[project]")
            (Path(tmpdir) / "package.json").write_text("{}")

            orientation = initializer.orient_to_codebase(tmpdir)

            assert "pyproject.toml" in orientation.config_files
            assert "package.json" in orientation.config_files

    def test_get_rules_of_engagement_no_alma(self, initializer):
        """Test getting rules without ALMA."""
        rules = initializer.get_rules_of_engagement("Helena")

        assert isinstance(rules, RulesOfEngagement)
        # Empty rules without ALMA
        assert len(rules.scope_rules) == 0

    def test_initialize_basic(self, initializer):
        """Test basic initialization."""
        result = initializer.initialize(
            project_id="test-project",
            agent="Helena",
            user_prompt="Test the login form",
        )

        assert result.project_id == "test-project"
        assert result.agent == "Helena"
        assert result.original_prompt == "Test the login form"
        assert len(result.work_items) >= 1

    def test_initialize_with_project_path(self, initializer):
        """Test initialization with project path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = initializer.initialize(
                project_id="proj",
                agent="Victor",
                user_prompt="Fix bug",
                project_path=tmpdir,
            )

            assert result.orientation is not None
            assert result.orientation.root_path == tmpdir

    def test_initialize_structured_prompt(self, initializer):
        """Test initialization with structured prompt."""
        result = initializer.initialize(
            project_id="proj",
            agent="Helena",
            user_prompt="""
            Test the following:
            - Login validation
            - Error messages
            - Success redirect
            """,
        )

        assert len(result.work_items) == 3
        assert result.recommended_start is not None

    def test_select_starting_point(self, initializer):
        """Test starting point selection."""
        from alma.progress import WorkItem

        items = [
            WorkItem.create("p", "Low priority", "desc", priority=10),
            WorkItem.create("p", "High priority", "desc", priority=90),
            WorkItem.create("p", "Medium priority", "desc", priority=50),
        ]

        start = initializer._select_starting_point(items)

        assert start is not None
        assert start.title == "High priority"

    def test_generate_orientation_summary(self, initializer):
        """Test orientation summary generation."""
        orientation = CodebaseOrientation(
            current_branch="develop",
            has_uncommitted_changes=True,
            recent_commits=["abc Initial"],
            root_path="/project",
            key_directories=["src", "tests"],
            config_files=["pyproject.toml"],
        )

        summary = initializer._generate_orientation_summary(orientation)

        assert "develop" in summary
        assert "uncommitted" in summary
        assert "Python project" in summary
