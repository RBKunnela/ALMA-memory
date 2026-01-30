"""
ALMA State Reducers.

Provides state reducers for merging parallel branch states in workflow
orchestration. Each reducer defines how to combine values from multiple
branches into a single value.

Sprint 1 Tasks 1.8-1.10
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

T = TypeVar("T")


class StateReducer(ABC):
    """
    Abstract base class for state reducers.

    A reducer defines how to combine multiple values (from parallel branches)
    into a single value during state merge operations.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of this reducer."""
        pass

    @abstractmethod
    def reduce(self, values: List[Any]) -> Any:
        """
        Reduce multiple values into a single value.

        Args:
            values: List of values from different branches.
                   May contain None values.

        Returns:
            The reduced single value.
        """
        pass


class AppendReducer(StateReducer):
    """
    Concatenates lists from all branches.

    Use for: messages, logs, notes, events
    """

    @property
    def name(self) -> str:
        return "append"

    def reduce(self, values: List[Any]) -> List[Any]:
        """Concatenate all lists, preserving order."""
        result: List[Any] = []
        for value in values:
            if value is None:
                continue
            if isinstance(value, list):
                result.extend(value)
            else:
                result.append(value)
        return result


class MergeDictReducer(StateReducer):
    """
    Merges dictionaries, with later values overwriting earlier ones.

    Use for: context, metadata, configuration
    """

    @property
    def name(self) -> str:
        return "merge_dict"

    def reduce(self, values: List[Any]) -> Dict[str, Any]:
        """Merge all dictionaries, later values win."""
        result: Dict[str, Any] = {}
        for value in values:
            if value is None:
                continue
            if isinstance(value, dict):
                result.update(value)
        return result


class LastValueReducer(StateReducer):
    """
    Takes the last non-None value.

    Use for: single values where the most recent is preferred
    """

    @property
    def name(self) -> str:
        return "last_value"

    def reduce(self, values: List[Any]) -> Any:
        """Return the last non-None value."""
        for value in reversed(values):
            if value is not None:
                return value
        return None


class FirstValueReducer(StateReducer):
    """
    Takes the first non-None value.

    Use for: priority values, initial state
    """

    @property
    def name(self) -> str:
        return "first_value"

    def reduce(self, values: List[Any]) -> Any:
        """Return the first non-None value."""
        for value in values:
            if value is not None:
                return value
        return None


class SumReducer(StateReducer):
    """
    Sums numeric values.

    Use for: counters, scores, totals
    """

    @property
    def name(self) -> str:
        return "sum"

    def reduce(self, values: List[Any]) -> Union[int, float]:
        """Sum all numeric values."""
        total: Union[int, float] = 0
        for value in values:
            if value is not None and isinstance(value, (int, float)):
                total += value
        return total


class MaxReducer(StateReducer):
    """
    Takes the maximum value.

    Use for: high scores, limits, thresholds
    """

    @property
    def name(self) -> str:
        return "max"

    def reduce(self, values: List[Any]) -> Optional[Union[int, float]]:
        """Return the maximum value."""
        numeric_values = [
            v for v in values if v is not None and isinstance(v, (int, float))
        ]
        return max(numeric_values) if numeric_values else None


class MinReducer(StateReducer):
    """
    Takes the minimum value.

    Use for: low scores, minimums
    """

    @property
    def name(self) -> str:
        return "min"

    def reduce(self, values: List[Any]) -> Optional[Union[int, float]]:
        """Return the minimum value."""
        numeric_values = [
            v for v in values if v is not None and isinstance(v, (int, float))
        ]
        return min(numeric_values) if numeric_values else None


class UnionReducer(StateReducer):
    """
    Creates a set union of all values.

    Use for: tags, categories, unique items
    """

    @property
    def name(self) -> str:
        return "union"

    def reduce(self, values: List[Any]) -> List[Any]:
        """Return union of all values as a list."""
        seen: set = set()
        result: List[Any] = []
        for value in values:
            if value is None:
                continue
            items = value if isinstance(value, (list, set)) else [value]
            for item in items:
                # Handle unhashable types
                try:
                    if item not in seen:
                        seen.add(item)
                        result.append(item)
                except TypeError:
                    # Unhashable type - just append
                    result.append(item)
        return result


# Built-in reducer instances
BUILTIN_REDUCERS: Dict[str, StateReducer] = {
    "append": AppendReducer(),
    "merge_dict": MergeDictReducer(),
    "last_value": LastValueReducer(),
    "first_value": FirstValueReducer(),
    "sum": SumReducer(),
    "max": MaxReducer(),
    "min": MinReducer(),
    "union": UnionReducer(),
}


def get_reducer(name: str) -> StateReducer:
    """
    Get a built-in reducer by name.

    Args:
        name: The reducer name.

    Returns:
        The reducer instance.

    Raises:
        ValueError: If reducer name is unknown.
    """
    if name not in BUILTIN_REDUCERS:
        raise ValueError(
            f"Unknown reducer: '{name}'. "
            f"Available reducers: {list(BUILTIN_REDUCERS.keys())}"
        )
    return BUILTIN_REDUCERS[name]


@dataclass
class ReducerConfig:
    """
    Configuration for state merging.

    Specifies which reducer to use for each field in the state.

    Attributes:
        field_reducers: Mapping of field names to reducer names.
        default_reducer: Default reducer for fields not in field_reducers.
        custom_reducers: Custom reducer instances by name.
    """

    field_reducers: Dict[str, str] = field(default_factory=dict)
    default_reducer: str = "last_value"
    custom_reducers: Dict[str, StateReducer] = field(default_factory=dict)

    def get_reducer_for_field(self, field_name: str) -> StateReducer:
        """
        Get the reducer for a specific field.

        Args:
            field_name: The field name.

        Returns:
            The reducer to use for this field.
        """
        reducer_name = self.field_reducers.get(field_name, self.default_reducer)

        # Check custom reducers first
        if reducer_name in self.custom_reducers:
            return self.custom_reducers[reducer_name]

        # Fall back to built-in
        return get_reducer(reducer_name)


class StateMerger:
    """
    Merges states from multiple parallel branches.

    Uses ReducerConfig to determine how each field should be merged.
    """

    def __init__(self, config: Optional[ReducerConfig] = None):
        """
        Initialize the state merger.

        Args:
            config: Reducer configuration. Uses defaults if not provided.
        """
        self.config = config or ReducerConfig()

    def merge(self, states: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge multiple states into a single state.

        Args:
            states: List of state dictionaries from parallel branches.

        Returns:
            The merged state dictionary.
        """
        if not states:
            return {}

        if len(states) == 1:
            return states[0].copy()

        # Collect all field names
        all_fields: set = set()
        for state in states:
            all_fields.update(state.keys())

        # Merge each field
        result: Dict[str, Any] = {}
        for field_name in all_fields:
            # Collect values for this field from all states
            values = [state.get(field_name) for state in states]

            # Get the appropriate reducer
            reducer = self.config.get_reducer_for_field(field_name)

            # Apply the reducer
            result[field_name] = reducer.reduce(values)

        return result

    def merge_checkpoints(
        self,
        checkpoints: List["Checkpoint"],  # type: ignore
    ) -> Dict[str, Any]:
        """
        Merge states from multiple checkpoints.

        Convenience method for merging checkpoint states.

        Args:
            checkpoints: List of Checkpoint objects.

        Returns:
            The merged state dictionary.
        """
        states = [cp.state for cp in checkpoints]
        return self.merge(states)


# Convenience function for simple merges
def merge_states(
    states: List[Dict[str, Any]],
    config: Optional[ReducerConfig] = None,
) -> Dict[str, Any]:
    """
    Merge multiple states into a single state.

    Convenience function that creates a StateMerger and merges.

    Args:
        states: List of state dictionaries.
        config: Optional reducer configuration.

    Returns:
        The merged state dictionary.

    Example:
        >>> config = ReducerConfig(
        ...     field_reducers={
        ...         "messages": "append",
        ...         "context": "merge_dict",
        ...         "total_score": "sum",
        ...     },
        ...     default_reducer="last_value",
        ... )
        >>> result = merge_states([state1, state2, state3], config)
    """
    merger = StateMerger(config)
    return merger.merge(states)
