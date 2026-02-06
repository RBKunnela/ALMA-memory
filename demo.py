#!/usr/bin/env python3
"""
ALMA Interactive Demo

Run this to see ALMA in action without any setup.
Uses in-memory storage and mock embeddings.

Usage:
    python demo.py
"""

from datetime import datetime


def print_header(text: str):
    """Print a section header."""
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print("=" * 60 + "\n")


def print_memory(label: str, items: list, formatter):
    """Print memories in a formatted way."""
    if not items:
        print(f"  {label}: (none)")
        return
    print(f"  {label}:")
    for item in items:
        print(f"    • {formatter(item)}")


def run_demo():
    """Run the interactive ALMA demo."""

    print_header("ALMA - Agent Learning Memory Architecture")
    print("This demo shows how ALMA helps AI agents learn from experience.\n")
    print("Unlike simple memory systems that just store facts,")
    print("ALMA learns STRATEGIES from outcomes - what works and what doesn't.\n")

    # =========================================================================
    # SETUP
    # =========================================================================
    print_header("Step 1: Initialize ALMA")

    print("Creating ALMA with in-memory storage (no database needed)...\n")

    # Inline minimal ALMA for demo (no external dependencies)
    import hashlib
    from dataclasses import dataclass, field
    from datetime import timezone
    from typing import Any, Dict, List

    @dataclass
    class Heuristic:
        id: str
        agent: str
        condition: str
        strategy: str
        confidence: float = 0.0
        occurrence_count: int = 0
        success_count: int = 0

    @dataclass
    class AntiPattern:
        id: str
        agent: str
        pattern: str
        why_bad: str
        better_alternative: str = ""
        occurrence_count: int = 1

    @dataclass
    class Outcome:
        id: str
        agent: str
        task: str
        success: bool
        strategy_used: str
        timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    class DemoALMA:
        """Minimal ALMA implementation for demo purposes."""

        def __init__(self):
            self.heuristics: List[Heuristic] = []
            self.anti_patterns: List[AntiPattern] = []
            self.outcomes: List[Outcome] = []
            self.agents = {
                "helena": {
                    "domain": "frontend_testing",
                    "can_learn": ["ui_testing", "form_validation"],
                },
                "victor": {
                    "domain": "backend_testing",
                    "can_learn": ["api_testing", "database"],
                },
            }

        def learn(
            self, agent: str, task: str, outcome: str, strategy: str, error: str = None
        ):
            """Learn from a task outcome."""
            success = outcome == "success"

            # Store outcome
            outcome_id = hashlib.md5(
                f"{task}{strategy}{datetime.now()}".encode()
            ).hexdigest()[:8]
            self.outcomes.append(
                Outcome(
                    id=outcome_id,
                    agent=agent,
                    task=task,
                    success=success,
                    strategy_used=strategy,
                )
            )

            # Check if we should create a heuristic (pattern seen 2+ times)
            similar_successes = [
                o
                for o in self.outcomes
                if o.agent == agent
                and o.success
                and strategy.lower() in o.strategy_used.lower()
            ]

            if success and len(similar_successes) >= 2:
                # Create or update heuristic
                existing = next(
                    (
                        h
                        for h in self.heuristics
                        if strategy.lower() in h.strategy.lower()
                    ),
                    None,
                )
                if existing:
                    existing.occurrence_count += 1
                    existing.success_count += 1
                    existing.confidence = (
                        existing.success_count / existing.occurrence_count
                    )
                else:
                    heuristic_id = hashlib.md5(
                        f"{agent}{strategy}".encode()
                    ).hexdigest()[:8]
                    self.heuristics.append(
                        Heuristic(
                            id=heuristic_id,
                            agent=agent,
                            condition=f"When testing similar to: {task[:50]}",
                            strategy=strategy,
                            confidence=0.7,
                            occurrence_count=len(similar_successes),
                            success_count=len(similar_successes),
                        )
                    )
                    return f"NEW HEURISTIC: '{strategy}' (worked {len(similar_successes)} times)"

            # Check if we should create anti-pattern
            if not success and error:
                similar_failures = [
                    o
                    for o in self.outcomes
                    if o.agent == agent
                    and not o.success
                    and strategy.lower() in o.strategy_used.lower()
                ]
                if len(similar_failures) >= 2:
                    existing_ap = next(
                        (
                            ap
                            for ap in self.anti_patterns
                            if strategy.lower() in ap.pattern.lower()
                        ),
                        None,
                    )
                    if not existing_ap:
                        ap_id = hashlib.md5(
                            f"{agent}{strategy}fail".encode()
                        ).hexdigest()[:8]
                        self.anti_patterns.append(
                            AntiPattern(
                                id=ap_id,
                                agent=agent,
                                pattern=strategy,
                                why_bad=error,
                            )
                        )
                        return f"NEW ANTI-PATTERN: Avoid '{strategy}' - {error}"

            return None

        def retrieve(self, task: str, agent: str) -> Dict[str, Any]:
            """Retrieve relevant memories for a task."""
            # Simple keyword matching for demo
            task_lower = task.lower()

            relevant_heuristics = [
                h
                for h in self.heuristics
                if h.agent == agent
                and any(
                    word in h.strategy.lower() or word in h.condition.lower()
                    for word in task_lower.split()
                )
            ]

            relevant_anti_patterns = [
                ap for ap in self.anti_patterns if ap.agent == agent
            ]

            recent_outcomes = [o for o in self.outcomes if o.agent == agent][-5:]

            return {
                "heuristics": relevant_heuristics,
                "anti_patterns": relevant_anti_patterns,
                "recent_outcomes": recent_outcomes,
            }

        def get_stats(self) -> Dict[str, int]:
            return {
                "heuristics": len(self.heuristics),
                "anti_patterns": len(self.anti_patterns),
                "outcomes": len(self.outcomes),
            }

    alma = DemoALMA()
    print("✓ ALMA initialized with 2 agents:")
    print("  • Helena (frontend testing specialist)")
    print("  • Victor (backend testing specialist)")

    # =========================================================================
    # SIMULATE LEARNING
    # =========================================================================
    print_header("Step 2: Simulate Agent Learning")

    print("Let's simulate Helena (QA agent) running some tests...\n")

    # Simulate multiple test runs
    test_scenarios = [
        {
            "task": "Test login form validation",
            "outcome": "success",
            "strategy": "Test each field incrementally, starting with empty submission",
            "error": None,
        },
        {
            "task": "Test registration form",
            "outcome": "success",
            "strategy": "Test each field incrementally, checking validation messages",
            "error": None,
        },
        {
            "task": "Test checkout flow",
            "outcome": "failure",
            "strategy": "Use sleep(5) to wait for page load",
            "error": "Flaky test - timing issues",
        },
        {
            "task": "Test payment form",
            "outcome": "failure",
            "strategy": "Use sleep(3) to wait for API response",
            "error": "Flaky test - race condition",
        },
        {
            "task": "Test password reset form",
            "outcome": "success",
            "strategy": "Test each field incrementally with explicit waits",
            "error": None,
        },
    ]

    for i, scenario in enumerate(test_scenarios, 1):
        print(f"Run #{i}: {scenario['task']}")
        print(f"  Strategy: {scenario['strategy'][:50]}...")
        print(f"  Outcome: {scenario['outcome'].upper()}")

        result = alma.learn(
            agent="helena",
            task=scenario["task"],
            outcome=scenario["outcome"],
            strategy=scenario["strategy"],
            error=scenario["error"],
        )

        if result:
            print(f"  📚 LEARNED: {result}")
        print()

    # =========================================================================
    # SHOW WHAT WAS LEARNED
    # =========================================================================
    print_header("Step 3: What Helena Learned")

    stats = alma.get_stats()
    print(
        f"Total memories: {stats['heuristics']} heuristics, {stats['anti_patterns']} anti-patterns\n"
    )

    print("HEURISTICS (what works):")
    for h in alma.heuristics:
        print(f"  ✓ {h.strategy}")
        print(
            f"    Confidence: {h.confidence:.0%} ({h.success_count}/{h.occurrence_count} successes)"
        )
        print()

    print("ANTI-PATTERNS (what to avoid):")
    for ap in alma.anti_patterns:
        print(f"  ✗ AVOID: {ap.pattern}")
        print(f"    Why: {ap.why_bad}")
        print()

    # =========================================================================
    # SHOW RETRIEVAL
    # =========================================================================
    print_header("Step 4: Memory Retrieval for New Task")

    new_task = "Test the contact form validation"
    print(f"New task: '{new_task}'\n")
    print("Retrieving relevant memories for Helena...\n")

    memories = alma.retrieve(new_task, "helena")

    print("RELEVANT HEURISTICS:")
    if memories["heuristics"]:
        for h in memories["heuristics"]:
            print(f"  → {h.strategy}")
    else:
        print("  (none found)")

    print("\nWARNINGS (anti-patterns to avoid):")
    if memories["anti_patterns"]:
        for ap in memories["anti_patterns"]:
            print(f"  ⚠ Don't use: {ap.pattern}")
            print(f"    Because: {ap.why_bad}")
    else:
        print("  (none)")

    # =========================================================================
    # SHOW PROMPT INJECTION
    # =========================================================================
    print_header("Step 5: Injecting Memories into Agent Prompt")

    print("Here's how ALMA enriches the agent's prompt:\n")
    print("-" * 50)

    prompt = f"""## Your Task
{new_task}

## What Has Worked Before
"""
    for h in memories["heuristics"]:
        prompt += f"- {h.strategy} (confidence: {h.confidence:.0%})\n"

    prompt += "\n## What to Avoid\n"
    for ap in memories["anti_patterns"]:
        prompt += f"- DON'T: {ap.pattern}\n  Reason: {ap.why_bad}\n"

    prompt += "\nNow execute the task using the knowledge above."

    print(prompt)
    print("-" * 50)

    # =========================================================================
    # KEY DIFFERENTIATOR
    # =========================================================================
    print_header("Why This Matters")

    print("""
ALMA vs Simple Memory (like Mem0):

┌─────────────────────────────────────────────────────────────┐
│  SIMPLE MEMORY                │  ALMA                       │
├─────────────────────────────────────────────────────────────┤
│  Stores: "User is vegetarian" │  Learns: "Incremental       │
│                               │  validation works for forms"│
├─────────────────────────────────────────────────────────────┤
│  Remembers FACTS              │  Learns STRATEGIES          │
├─────────────────────────────────────────────────────────────┤
│  Passive storage              │  Active learning            │
├─────────────────────────────────────────────────────────────┤
│  Same for all agents          │  Scoped per agent domain    │
├─────────────────────────────────────────────────────────────┤
│  No anti-patterns             │  Tracks what NOT to do      │
└─────────────────────────────────────────────────────────────┘

ALMA doesn't just remember - it LEARNS from experience.
""")

    print_header("Try ALMA")
    print("GitHub: https://github.com/RBKunnela/ALMA-memory")
    print("Install: pip install alma-memory")
    print("\nQuestions? Open an issue or DM @RBKunnela on LinkedIn")


if __name__ == "__main__":
    run_demo()
