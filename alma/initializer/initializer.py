"""
Session Initializer.

Bootstrap pattern that orients the agent before work begins.
"Stage manager sets the stage, actor performs."
"""

import logging
import re
import subprocess
from pathlib import Path
from typing import Any, List, Optional

from alma.initializer.types import (
    CodebaseOrientation,
    InitializationResult,
    RulesOfEngagement,
)

logger = logging.getLogger(__name__)


class SessionInitializer:
    """
    Bootstrap domain memory from user prompt.

    The Initializer Pattern:
    1. Expand user prompt to structured work items
    2. Orient to current codebase state (git, files)
    3. Retrieve relevant memories from past sessions
    4. Set rules of engagement from agent scope
    5. Suggest optimal starting point

    Usage:
        initializer = SessionInitializer(alma)

        result = initializer.initialize(
            project_id="my-project",
            agent="Helena",
            user_prompt="Test the login flow",
        )

        # Inject into agent prompt
        prompt = f'''
        {result.to_prompt()}

        Now proceed with the first work item.
        '''
    """

    def __init__(
        self,
        alma: Optional[Any] = None,
        progress_tracker: Optional[Any] = None,
        session_manager: Optional[Any] = None,
    ):
        """
        Initialize the SessionInitializer.

        Args:
            alma: ALMA instance for memory retrieval
            progress_tracker: ProgressTracker for work item management
            session_manager: SessionManager for session context
        """
        self.alma = alma
        self.progress_tracker = progress_tracker
        self.session_manager = session_manager

    def initialize(
        self,
        project_id: str,
        agent: str,
        user_prompt: str,
        project_path: Optional[str] = None,
        auto_expand: bool = True,
        memory_top_k: int = 5,
    ) -> InitializationResult:
        """
        Full session initialization.

        Args:
            project_id: Project identifier
            agent: Agent name (e.g., "Helena", "Victor")
            user_prompt: Raw user prompt/task
            project_path: Optional path to project root (for git orientation)
            auto_expand: Whether to expand prompt to work items
            memory_top_k: How many memories to retrieve

        Returns:
            InitializationResult with everything agent needs
        """
        logger.info(f"Initializing session for {agent} on {project_id}")

        # Create result
        result = InitializationResult.create(
            project_id=project_id,
            agent=agent,
            original_prompt=user_prompt,
        )

        # 1. Expand prompt to work items
        if auto_expand:
            work_items = self.expand_prompt(user_prompt)
            result.work_items = work_items
            if work_items:
                result.goal = self._summarize_goal(user_prompt, work_items)

        # 2. Orient to codebase
        if project_path:
            result.orientation = self.orient_to_codebase(project_path)

        # 3. Retrieve relevant memories
        if self.alma:
            try:
                memories = self.alma.retrieve(
                    task=user_prompt,
                    agent=agent,
                    top_k=memory_top_k,
                )
                result.relevant_memories = memories
            except Exception as e:
                logger.warning(f"Failed to retrieve memories: {e}")

        # 4. Get rules of engagement
        if self.alma:
            result.rules = self.get_rules_of_engagement(agent)

        # 5. Suggest starting point
        if result.work_items:
            result.recommended_start = self._select_starting_point(result.work_items)

        # 6. Get recent activity from session manager
        if self.session_manager:
            try:
                context = self.session_manager.start_session(
                    agent=agent,
                    goal=result.goal,
                )
                result.session_id = context.session_id
                if context.previous_handoff:
                    result.recent_activity = context.previous_handoff.next_steps or []
            except Exception as e:
                logger.warning(f"Failed to get session context: {e}")

        logger.info(
            f"Initialization complete: {len(result.work_items)} work items, "
            f"orientation: {'yes' if result.orientation else 'no'}, "
            f"memories: {'yes' if result.relevant_memories else 'no'}"
        )

        return result

    def expand_prompt(
        self,
        user_prompt: str,
        use_ai: bool = False,
    ) -> List[Any]:
        """
        Expand user prompt into structured work items.

        Simple implementation: extract bullet points and numbered items.
        AI implementation: use LLM to break down complex tasks.

        Args:
            user_prompt: Raw user prompt
            use_ai: Whether to use AI for expansion (requires LLM)

        Returns:
            List of WorkItem objects
        """
        from alma.progress import WorkItem

        work_items = []

        # Simple extraction: look for bullet points and numbered items
        lines = user_prompt.strip().split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Match bullet points: -, *, •
            bullet_match = re.match(r"^[-*•]\s+(.+)$", line)
            if bullet_match:
                title = bullet_match.group(1).strip()
                work_items.append(
                    WorkItem.create(
                        project_id="",  # Will be set by caller
                        title=title,
                        description=title,
                    )
                )
                continue

            # Match numbered items: 1., 2., etc.
            number_match = re.match(r"^\d+\.\s+(.+)$", line)
            if number_match:
                title = number_match.group(1).strip()
                work_items.append(
                    WorkItem.create(
                        project_id="",
                        title=title,
                        description=title,
                    )
                )
                continue

        # If no structured items found, create single item from prompt
        if not work_items:
            # Truncate long prompts for title
            title = user_prompt[:100].strip()
            if len(user_prompt) > 100:
                title += "..."

            work_items.append(
                WorkItem.create(
                    project_id="",
                    title=title,
                    description=user_prompt,
                )
            )

        return work_items

    def orient_to_codebase(
        self,
        project_path: str,
        max_commits: int = 5,
    ) -> CodebaseOrientation:
        """
        Orient to current codebase state.

        Reads git status, recent commits, and file structure.

        Args:
            project_path: Path to project root
            max_commits: Max number of recent commits to include

        Returns:
            CodebaseOrientation with codebase state
        """
        path = Path(project_path)

        # Default orientation
        orientation = CodebaseOrientation(
            current_branch="unknown",
            has_uncommitted_changes=False,
            recent_commits=[],
            root_path=str(path),
            key_directories=[],
            config_files=[],
        )

        # Check if it's a git repo
        git_dir = path / ".git"
        is_git_repo = git_dir.exists()

        if is_git_repo:
            try:
                # Get current branch
                result = subprocess.run(
                    ["git", "branch", "--show-current"],
                    cwd=path,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    orientation.current_branch = (
                        result.stdout.strip() or "HEAD detached"
                    )

                # Check for uncommitted changes
                result = subprocess.run(
                    ["git", "status", "--porcelain"],
                    cwd=path,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    orientation.has_uncommitted_changes = bool(result.stdout.strip())

                # Get recent commits
                result = subprocess.run(
                    ["git", "log", "--oneline", f"-{max_commits}"],
                    cwd=path,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    commits = result.stdout.strip().split("\n")
                    orientation.recent_commits = [c for c in commits if c]

            except subprocess.TimeoutExpired:
                logger.warning("Git commands timed out")
            except Exception as e:
                logger.warning(f"Git orientation failed: {e}")

        # Find key directories
        key_dirs = ["src", "lib", "tests", "test", "app", "api", "core"]
        orientation.key_directories = [d for d in key_dirs if (path / d).is_dir()]

        # Find config files
        config_files = [
            "package.json",
            "pyproject.toml",
            "setup.py",
            "Cargo.toml",
            "go.mod",
            "pom.xml",
            "build.gradle",
            "Makefile",
            "CMakeLists.txt",
        ]
        orientation.config_files = [f for f in config_files if (path / f).exists()]

        # Generate summary
        orientation.summary = self._generate_orientation_summary(orientation)

        return orientation

    def get_rules_of_engagement(
        self,
        agent: str,
    ) -> RulesOfEngagement:
        """
        Get rules of engagement from agent scope.

        Args:
            agent: Agent name

        Returns:
            RulesOfEngagement with scope rules, constraints, quality gates
        """
        rules = RulesOfEngagement()

        if not self.alma:
            return rules

        # Get scope from ALMA
        scope = self.alma.scopes.get(agent)
        if not scope:
            return rules

        # Convert scope to rules
        if scope.can_learn:
            rules.scope_rules = [f"Learn from: {', '.join(scope.can_learn)}"]

        if scope.cannot_learn:
            rules.constraints = [f"Do not learn from: {', '.join(scope.cannot_learn)}"]

        # Default quality gates
        rules.quality_gates = [
            "All tests pass",
            "No regressions introduced",
            "Changes documented if significant",
        ]

        return rules

    def _summarize_goal(self, prompt: str, work_items: List[Any]) -> str:
        """Summarize goal from prompt and work items."""
        if len(work_items) == 1:
            return prompt

        item_titles = [getattr(item, "title", str(item)) for item in work_items]
        return f"{prompt}\n\nBroken down into {len(work_items)} items: {', '.join(item_titles[:3])}{'...' if len(item_titles) > 3 else ''}"

    def _select_starting_point(self, work_items: List[Any]) -> Optional[Any]:
        """Select the best starting point from work items."""
        if not work_items:
            return None

        # Find highest priority unblocked item
        actionable = [
            item
            for item in work_items
            if getattr(item, "status", "pending") == "pending"
            and not getattr(item, "blocked_by", [])
        ]

        if actionable:
            # Sort by priority (higher = more important)
            actionable.sort(key=lambda x: getattr(x, "priority", 50), reverse=True)
            return actionable[0]

        return work_items[0]

    def _generate_orientation_summary(self, orientation: CodebaseOrientation) -> str:
        """Generate a one-line summary of codebase orientation."""
        parts = []

        parts.append(f"Branch: {orientation.current_branch}")

        if orientation.has_uncommitted_changes:
            parts.append("has uncommitted changes")

        if orientation.key_directories:
            parts.append(f"key dirs: {', '.join(orientation.key_directories[:3])}")

        if orientation.config_files:
            # Infer project type from config files
            if "package.json" in orientation.config_files:
                parts.append("Node.js project")
            elif (
                "pyproject.toml" in orientation.config_files
                or "setup.py" in orientation.config_files
            ):
                parts.append("Python project")
            elif "Cargo.toml" in orientation.config_files:
                parts.append("Rust project")
            elif "go.mod" in orientation.config_files:
                parts.append("Go project")

        return "; ".join(parts)
