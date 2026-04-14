"""
Task Dependency DAG Generator.

Generates test DAGs with known structures for benchmarking task dependency
resolution. Each shape has predictable properties that make it useful for
testing specific aspects of dependency tracking:

- chain: linear dependency, tests sequential ordering
- diamond: convergence pattern, tests multi-parent resolution
- wide_fan: one root with many leaves, tests parallel readiness
- deep_narrow: tall skinny tree, tests depth handling
- forest: independent chains, tests isolation between subgraphs
- cyclic: deliberately invalid, tests cycle rejection
"""

from typing import Dict, List, Set, Tuple


def generate_dag(
    shape: str,
    n: int,
) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Generate a test DAG with known structure.

    Args:
        shape: One of ``'chain'``, ``'diamond'``, ``'wide_fan'``,
            ``'deep_narrow'``, ``'forest'``, ``'cyclic'``.
        n: Number of tasks. Minimum depends on shape (e.g., diamond
            needs at least 4). Values below the minimum are clamped up.

    Returns:
        Tuple of ``(task_ids, edges)`` where ``edges`` are
        ``(from_id, to_id)`` dependency pairs meaning ``to_id``
        depends on ``from_id`` (``from_id`` must complete before
        ``to_id`` can start).

    Raises:
        ValueError: If shape is not recognized.

    Example:
        >>> ids, edges = generate_dag("chain", 4)
        >>> ids
        ['t0', 't1', 't2', 't3']
        >>> edges
        [('t0', 't1'), ('t1', 't2'), ('t2', 't3')]
    """
    generators = {
        "chain": _chain,
        "diamond": _diamond,
        "wide_fan": _wide_fan,
        "deep_narrow": _deep_narrow,
        "forest": _forest,
        "cyclic": _cyclic,
    }

    if shape not in generators:
        raise ValueError(
            f"Unknown shape '{shape}'. Valid shapes: {sorted(generators.keys())}"
        )

    return generators[shape](n)


def _chain(n: int) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Linear chain: t0 -> t1 -> t2 -> ... -> t(n-1).

    Every task depends on the previous one. Only t0 is initially ready.
    """
    n = max(n, 2)
    task_ids = [f"t{i}" for i in range(n)]
    edges = [(f"t{i}", f"t{i + 1}") for i in range(n - 1)]
    return task_ids, edges


def _diamond(n: int) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Diamond pattern: one root fans out to middle tasks, which converge to a sink.

    For n=4: t0 -> {t1, t2} -> t3
    For larger n: t0 -> {t1, ..., t(n-2)} -> t(n-1)
    The middle layer has (n - 2) tasks that can all run in parallel.
    """
    n = max(n, 4)
    task_ids = [f"t{i}" for i in range(n)]
    edges: List[Tuple[str, str]] = []

    root = "t0"
    sink = f"t{n - 1}"

    # Root fans out to middle layer
    for i in range(1, n - 1):
        edges.append((root, f"t{i}"))

    # Middle layer converges to sink
    for i in range(1, n - 1):
        edges.append((f"t{i}", sink))

    return task_ids, edges


def _wide_fan(n: int) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Wide fan: one root, (n-1) leaf tasks.

    t0 -> {t1, t2, ..., t(n-1)}
    After t0 completes, all other tasks become ready simultaneously.
    Tests parallel readiness detection.
    """
    n = max(n, 2)
    task_ids = [f"t{i}" for i in range(n)]
    edges = [("t0", f"t{i}") for i in range(1, n)]
    return task_ids, edges


def _deep_narrow(n: int) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Deep narrow tree: depth = n, width = 2 at each level.

    Each level has two tasks. The left task of each level depends on
    the left task of the previous level. The right task depends on
    the right task of the previous level. This creates two parallel
    chains of depth n.

    Total tasks: 2 * n.
    """
    n = max(n, 2)
    task_ids: List[str] = []
    edges: List[Tuple[str, str]] = []

    for level in range(n):
        left = f"t{level * 2}"
        right = f"t{level * 2 + 1}"
        task_ids.extend([left, right])

        if level > 0:
            prev_left = f"t{(level - 1) * 2}"
            prev_right = f"t{(level - 1) * 2 + 1}"
            edges.append((prev_left, left))
            edges.append((prev_right, right))

    return task_ids, edges


def _forest(n: int) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Forest: 10 independent linear chains.

    Each chain has n // 10 tasks (minimum 2). Chains are completely
    independent -- completing tasks in one chain has no effect on others.
    Tests isolation between subgraphs.

    Total tasks: 10 * chain_length.
    """
    num_chains = 10
    chain_len = max(n // num_chains, 2)

    task_ids: List[str] = []
    edges: List[Tuple[str, str]] = []

    for chain_idx in range(num_chains):
        prefix = f"c{chain_idx}"
        for pos in range(chain_len):
            task_id = f"{prefix}_t{pos}"
            task_ids.append(task_id)
            if pos > 0:
                prev_id = f"{prefix}_t{pos - 1}"
                edges.append((prev_id, task_id))

    return task_ids, edges


def _cyclic(n: int) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Cyclic graph: deliberately creates a cycle for rejection testing.

    Creates a chain t0 -> t1 -> ... -> t(n-1) and adds a back-edge
    from the last task to the first, creating a cycle. Any correct
    dependency resolver should detect and reject this.
    """
    n = max(n, 3)
    task_ids = [f"t{i}" for i in range(n)]
    edges = [(f"t{i}", f"t{i + 1}") for i in range(n - 1)]
    # Add the cycle-creating back-edge
    edges.append((f"t{n - 1}", "t0"))
    return task_ids, edges


def get_expected_ready_tasks(
    task_ids: List[str],
    edges: List[Tuple[str, str]],
    completed: Set[str],
) -> Set[str]:
    """Given completed tasks, return the set of tasks that should be unblocked.

    A task is ready when:
    1. It has not been completed yet.
    2. All of its dependencies (predecessors in the edge list) have been completed.

    Tasks with no dependencies are immediately ready (if not completed).

    Args:
        task_ids: All task IDs in the DAG.
        edges: Dependency edges as ``(from_id, to_id)`` pairs.
        completed: Set of task IDs that have been completed.

    Returns:
        Set of task IDs that are ready to execute.

    Example:
        >>> ids, edges = generate_dag("diamond", 4)
        >>> get_expected_ready_tasks(ids, edges, set())
        {'t0'}
        >>> get_expected_ready_tasks(ids, edges, {'t0'})
        {'t1', 't2'}
        >>> get_expected_ready_tasks(ids, edges, {'t0', 't1', 't2'})
        {'t3'}
    """
    # Build dependency map: task -> set of tasks it depends on
    deps: Dict[str, Set[str]] = {tid: set() for tid in task_ids}
    for from_id, to_id in edges:
        if to_id in deps:
            deps[to_id].add(from_id)

    ready: Set[str] = set()
    for tid in task_ids:
        if tid in completed:
            continue
        # Ready if all dependencies are completed
        if deps[tid].issubset(completed):
            ready.add(tid)

    return ready
