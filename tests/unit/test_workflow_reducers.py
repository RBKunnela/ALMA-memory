"""
Unit tests for ALMA State Reducers.

Tests for:
- StateReducer abstract class (Task 1.12)
- All built-in reducers (append, merge_dict, last_value, first_value, sum, max, min, union)
- ReducerConfig (Task 1.12)
- StateMerger (Task 1.12)

Sprint 1 Task 1.12
"""

import pytest

from alma.workflow.checkpoint import Checkpoint
from alma.workflow.reducers import (
    BUILTIN_REDUCERS,
    AppendReducer,
    FirstValueReducer,
    LastValueReducer,
    MaxReducer,
    MergeDictReducer,
    MinReducer,
    ReducerConfig,
    StateMerger,
    StateReducer,
    SumReducer,
    UnionReducer,
    get_reducer,
    merge_states,
)

# =============================================================================
# StateReducer Abstract Class Tests
# =============================================================================


class TestStateReducerInterface:
    """Tests for StateReducer interface."""

    def test_builtin_reducers_registered(self):
        """Test all built-in reducers are registered."""
        assert "append" in BUILTIN_REDUCERS
        assert "merge_dict" in BUILTIN_REDUCERS
        assert "last_value" in BUILTIN_REDUCERS
        assert "first_value" in BUILTIN_REDUCERS
        assert "sum" in BUILTIN_REDUCERS
        assert "max" in BUILTIN_REDUCERS
        assert "min" in BUILTIN_REDUCERS
        assert "union" in BUILTIN_REDUCERS

    def test_get_reducer_valid(self):
        """Test getting valid reducers."""
        reducer = get_reducer("append")
        assert isinstance(reducer, AppendReducer)

        reducer = get_reducer("sum")
        assert isinstance(reducer, SumReducer)

    def test_get_reducer_invalid(self):
        """Test getting invalid reducer raises error."""
        with pytest.raises(ValueError) as exc:
            get_reducer("invalid_reducer")
        assert "Unknown reducer" in str(exc.value)
        assert "invalid_reducer" in str(exc.value)

    def test_all_reducers_have_name(self):
        """Test all reducers implement name property."""
        for name, reducer in BUILTIN_REDUCERS.items():
            assert reducer.name == name

    def test_all_reducers_have_reduce(self):
        """Test all reducers implement reduce method."""
        for reducer in BUILTIN_REDUCERS.values():
            # Should not raise - even with empty list
            reducer.reduce([])


# =============================================================================
# AppendReducer Tests
# =============================================================================


class TestAppendReducer:
    """Tests for AppendReducer."""

    @pytest.fixture
    def reducer(self):
        return AppendReducer()

    def test_name(self, reducer):
        """Test reducer name."""
        assert reducer.name == "append"

    def test_append_lists(self, reducer):
        """Test appending multiple lists."""
        values = [[1, 2], [3, 4], [5, 6]]
        result = reducer.reduce(values)
        assert result == [1, 2, 3, 4, 5, 6]

    def test_append_single_values(self, reducer):
        """Test appending single values (not lists)."""
        values = [1, 2, 3]
        result = reducer.reduce(values)
        assert result == [1, 2, 3]

    def test_append_mixed(self, reducer):
        """Test appending mixed lists and single values."""
        values = [[1, 2], 3, [4, 5]]
        result = reducer.reduce(values)
        assert result == [1, 2, 3, 4, 5]

    def test_append_with_none(self, reducer):
        """Test appending with None values."""
        values = [[1, 2], None, [3, 4]]
        result = reducer.reduce(values)
        assert result == [1, 2, 3, 4]

    def test_append_all_none(self, reducer):
        """Test appending all None values."""
        values = [None, None, None]
        result = reducer.reduce(values)
        assert result == []

    def test_append_empty(self, reducer):
        """Test appending empty list."""
        result = reducer.reduce([])
        assert result == []

    def test_append_strings(self, reducer):
        """Test appending string lists."""
        values = [["a", "b"], ["c"], ["d", "e"]]
        result = reducer.reduce(values)
        assert result == ["a", "b", "c", "d", "e"]


# =============================================================================
# MergeDictReducer Tests
# =============================================================================


class TestMergeDictReducer:
    """Tests for MergeDictReducer."""

    @pytest.fixture
    def reducer(self):
        return MergeDictReducer()

    def test_name(self, reducer):
        """Test reducer name."""
        assert reducer.name == "merge_dict"

    def test_merge_dicts(self, reducer):
        """Test merging multiple dicts."""
        values = [{"a": 1}, {"b": 2}, {"c": 3}]
        result = reducer.reduce(values)
        assert result == {"a": 1, "b": 2, "c": 3}

    def test_merge_overlapping(self, reducer):
        """Test merging dicts with overlapping keys (later wins)."""
        values = [{"a": 1, "b": 2}, {"b": 3, "c": 4}]
        result = reducer.reduce(values)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_merge_with_none(self, reducer):
        """Test merging with None values."""
        values = [{"a": 1}, None, {"b": 2}]
        result = reducer.reduce(values)
        assert result == {"a": 1, "b": 2}

    def test_merge_all_none(self, reducer):
        """Test merging all None values."""
        values = [None, None]
        result = reducer.reduce(values)
        assert result == {}

    def test_merge_empty(self, reducer):
        """Test merging empty list."""
        result = reducer.reduce([])
        assert result == {}

    def test_merge_nested_dicts(self, reducer):
        """Test merging nested dicts (shallow merge)."""
        values = [
            {"nested": {"a": 1}},
            {"nested": {"b": 2}},  # Replaces, doesn't deep merge
        ]
        result = reducer.reduce(values)
        assert result == {"nested": {"b": 2}}

    def test_merge_non_dict_ignored(self, reducer):
        """Test non-dict values are ignored."""
        values = [{"a": 1}, "not a dict", {"b": 2}, 123]
        result = reducer.reduce(values)
        assert result == {"a": 1, "b": 2}


# =============================================================================
# LastValueReducer Tests
# =============================================================================


class TestLastValueReducer:
    """Tests for LastValueReducer."""

    @pytest.fixture
    def reducer(self):
        return LastValueReducer()

    def test_name(self, reducer):
        """Test reducer name."""
        assert reducer.name == "last_value"

    def test_last_value(self, reducer):
        """Test getting last non-None value."""
        values = [1, 2, 3]
        result = reducer.reduce(values)
        assert result == 3

    def test_last_value_with_none(self, reducer):
        """Test with trailing None values."""
        values = [1, 2, None, None]
        result = reducer.reduce(values)
        assert result == 2

    def test_last_value_mixed_none(self, reducer):
        """Test with mixed None values."""
        values = [1, None, 3, None]
        result = reducer.reduce(values)
        assert result == 3

    def test_last_value_all_none(self, reducer):
        """Test all None values."""
        values = [None, None, None]
        result = reducer.reduce(values)
        assert result is None

    def test_last_value_empty(self, reducer):
        """Test empty list."""
        result = reducer.reduce([])
        assert result is None

    def test_last_value_single(self, reducer):
        """Test single value."""
        result = reducer.reduce([42])
        assert result == 42


# =============================================================================
# FirstValueReducer Tests
# =============================================================================


class TestFirstValueReducer:
    """Tests for FirstValueReducer."""

    @pytest.fixture
    def reducer(self):
        return FirstValueReducer()

    def test_name(self, reducer):
        """Test reducer name."""
        assert reducer.name == "first_value"

    def test_first_value(self, reducer):
        """Test getting first non-None value."""
        values = [1, 2, 3]
        result = reducer.reduce(values)
        assert result == 1

    def test_first_value_with_leading_none(self, reducer):
        """Test with leading None values."""
        values = [None, None, 3, 4]
        result = reducer.reduce(values)
        assert result == 3

    def test_first_value_all_none(self, reducer):
        """Test all None values."""
        values = [None, None, None]
        result = reducer.reduce(values)
        assert result is None

    def test_first_value_empty(self, reducer):
        """Test empty list."""
        result = reducer.reduce([])
        assert result is None


# =============================================================================
# SumReducer Tests
# =============================================================================


class TestSumReducer:
    """Tests for SumReducer."""

    @pytest.fixture
    def reducer(self):
        return SumReducer()

    def test_name(self, reducer):
        """Test reducer name."""
        assert reducer.name == "sum"

    def test_sum_integers(self, reducer):
        """Test summing integers."""
        values = [1, 2, 3, 4, 5]
        result = reducer.reduce(values)
        assert result == 15

    def test_sum_floats(self, reducer):
        """Test summing floats."""
        values = [1.5, 2.5, 3.0]
        result = reducer.reduce(values)
        assert result == 7.0

    def test_sum_mixed(self, reducer):
        """Test summing mixed int/float."""
        values = [1, 2.5, 3]
        result = reducer.reduce(values)
        assert result == 6.5

    def test_sum_with_none(self, reducer):
        """Test summing with None values."""
        values = [1, None, 2, None, 3]
        result = reducer.reduce(values)
        assert result == 6

    def test_sum_all_none(self, reducer):
        """Test all None values."""
        values = [None, None]
        result = reducer.reduce(values)
        assert result == 0

    def test_sum_empty(self, reducer):
        """Test empty list."""
        result = reducer.reduce([])
        assert result == 0

    def test_sum_non_numeric_ignored(self, reducer):
        """Test non-numeric values are ignored."""
        values = [1, "string", 2, [3], 4]
        result = reducer.reduce(values)
        assert result == 7


# =============================================================================
# MaxReducer Tests
# =============================================================================


class TestMaxReducer:
    """Tests for MaxReducer."""

    @pytest.fixture
    def reducer(self):
        return MaxReducer()

    def test_name(self, reducer):
        """Test reducer name."""
        assert reducer.name == "max"

    def test_max_integers(self, reducer):
        """Test max of integers."""
        values = [1, 5, 3, 2, 4]
        result = reducer.reduce(values)
        assert result == 5

    def test_max_floats(self, reducer):
        """Test max of floats."""
        values = [1.5, 3.7, 2.1]
        result = reducer.reduce(values)
        assert result == 3.7

    def test_max_negative(self, reducer):
        """Test max with negative numbers."""
        values = [-5, -1, -10]
        result = reducer.reduce(values)
        assert result == -1

    def test_max_with_none(self, reducer):
        """Test max with None values."""
        values = [1, None, 5, None, 3]
        result = reducer.reduce(values)
        assert result == 5

    def test_max_all_none(self, reducer):
        """Test all None values."""
        values = [None, None]
        result = reducer.reduce(values)
        assert result is None

    def test_max_empty(self, reducer):
        """Test empty list."""
        result = reducer.reduce([])
        assert result is None

    def test_max_non_numeric_ignored(self, reducer):
        """Test non-numeric values are ignored."""
        values = [1, "string", 10, [100]]
        result = reducer.reduce(values)
        assert result == 10


# =============================================================================
# MinReducer Tests
# =============================================================================


class TestMinReducer:
    """Tests for MinReducer."""

    @pytest.fixture
    def reducer(self):
        return MinReducer()

    def test_name(self, reducer):
        """Test reducer name."""
        assert reducer.name == "min"

    def test_min_integers(self, reducer):
        """Test min of integers."""
        values = [5, 1, 3, 2, 4]
        result = reducer.reduce(values)
        assert result == 1

    def test_min_floats(self, reducer):
        """Test min of floats."""
        values = [1.5, 3.7, 0.5]
        result = reducer.reduce(values)
        assert result == 0.5

    def test_min_negative(self, reducer):
        """Test min with negative numbers."""
        values = [-5, -1, -10]
        result = reducer.reduce(values)
        assert result == -10

    def test_min_with_none(self, reducer):
        """Test min with None values."""
        values = [5, None, 1, None, 3]
        result = reducer.reduce(values)
        assert result == 1

    def test_min_all_none(self, reducer):
        """Test all None values."""
        values = [None, None]
        result = reducer.reduce(values)
        assert result is None

    def test_min_empty(self, reducer):
        """Test empty list."""
        result = reducer.reduce([])
        assert result is None


# =============================================================================
# UnionReducer Tests
# =============================================================================


class TestUnionReducer:
    """Tests for UnionReducer."""

    @pytest.fixture
    def reducer(self):
        return UnionReducer()

    def test_name(self, reducer):
        """Test reducer name."""
        assert reducer.name == "union"

    def test_union_lists(self, reducer):
        """Test union of lists."""
        values = [[1, 2], [2, 3], [3, 4]]
        result = reducer.reduce(values)
        assert set(result) == {1, 2, 3, 4}

    def test_union_sets(self, reducer):
        """Test union of sets."""
        values = [{1, 2}, {2, 3}, {3, 4}]
        result = reducer.reduce(values)
        assert set(result) == {1, 2, 3, 4}

    def test_union_single_values(self, reducer):
        """Test union of single values."""
        values = [1, 2, 2, 3]
        result = reducer.reduce(values)
        assert set(result) == {1, 2, 3}

    def test_union_mixed(self, reducer):
        """Test union of mixed types."""
        values = [[1, 2], 3, {4, 5}]
        result = reducer.reduce(values)
        assert set(result) == {1, 2, 3, 4, 5}

    def test_union_with_none(self, reducer):
        """Test union with None values."""
        values = [[1, 2], None, [3]]
        result = reducer.reduce(values)
        assert set(result) == {1, 2, 3}

    def test_union_all_none(self, reducer):
        """Test all None values."""
        values = [None, None]
        result = reducer.reduce(values)
        assert result == []

    def test_union_empty(self, reducer):
        """Test empty list."""
        result = reducer.reduce([])
        assert result == []

    def test_union_strings(self, reducer):
        """Test union of strings."""
        values = [["a", "b"], ["b", "c"]]
        result = reducer.reduce(values)
        assert set(result) == {"a", "b", "c"}

    def test_union_preserves_order(self, reducer):
        """Test union preserves first occurrence order."""
        values = [[1, 2, 3], [2, 4], [1, 5]]
        result = reducer.reduce(values)
        # First occurrences should be in order: 1, 2, 3, 4, 5
        assert result == [1, 2, 3, 4, 5]

    def test_union_unhashable_types(self, reducer):
        """Test union handles unhashable types (appends all)."""
        values = [[{"a": 1}], [{"b": 2}]]
        result = reducer.reduce(values)
        # Unhashable types are just appended
        assert len(result) == 2
        assert {"a": 1} in result
        assert {"b": 2} in result


# =============================================================================
# ReducerConfig Tests
# =============================================================================


class TestReducerConfig:
    """Tests for ReducerConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = ReducerConfig()
        assert config.field_reducers == {}
        assert config.default_reducer == "last_value"
        assert config.custom_reducers == {}

    def test_field_reducers(self):
        """Test field-specific reducers."""
        config = ReducerConfig(
            field_reducers={
                "messages": "append",
                "context": "merge_dict",
                "score": "sum",
            }
        )

        assert isinstance(config.get_reducer_for_field("messages"), AppendReducer)
        assert isinstance(config.get_reducer_for_field("context"), MergeDictReducer)
        assert isinstance(config.get_reducer_for_field("score"), SumReducer)

    def test_default_reducer_fallback(self):
        """Test default reducer is used for unspecified fields."""
        config = ReducerConfig(
            field_reducers={"messages": "append"},
            default_reducer="last_value",
        )

        # Specified field
        assert isinstance(config.get_reducer_for_field("messages"), AppendReducer)

        # Unspecified field uses default
        assert isinstance(config.get_reducer_for_field("other_field"), LastValueReducer)

    def test_custom_reducers(self):
        """Test custom reducer instances."""

        class DoubleReducer(StateReducer):
            @property
            def name(self):
                return "double"

            def reduce(self, values):
                total = sum(v for v in values if isinstance(v, (int, float)))
                return total * 2

        config = ReducerConfig(
            field_reducers={"score": "double"},
            custom_reducers={"double": DoubleReducer()},
        )

        reducer = config.get_reducer_for_field("score")
        assert isinstance(reducer, DoubleReducer)
        assert reducer.reduce([1, 2, 3]) == 12  # (1+2+3) * 2


# =============================================================================
# StateMerger Tests
# =============================================================================


class TestStateMerger:
    """Tests for StateMerger."""

    def test_default_merger(self):
        """Test merger with default config."""
        merger = StateMerger()
        assert merger.config is not None

    def test_merge_empty_states(self):
        """Test merging empty list."""
        merger = StateMerger()
        result = merger.merge([])
        assert result == {}

    def test_merge_single_state(self):
        """Test merging single state returns copy."""
        merger = StateMerger()
        state = {"a": 1, "b": 2}
        result = merger.merge([state])
        assert result == state
        assert result is not state  # Should be a copy

    def test_merge_multiple_states_default(self):
        """Test merging multiple states with default reducer."""
        merger = StateMerger()
        states = [
            {"a": 1, "b": 2},
            {"a": 3, "c": 4},
        ]
        result = merger.merge(states)
        # Default is last_value, so later values win
        assert result["a"] == 3
        assert result["b"] == 2
        assert result["c"] == 4

    def test_merge_with_custom_config(self):
        """Test merging with custom reducer config."""
        config = ReducerConfig(
            field_reducers={
                "messages": "append",
                "counts": "sum",
                "tags": "union",
            },
            default_reducer="last_value",
        )
        merger = StateMerger(config)

        states = [
            {
                "messages": ["hello"],
                "counts": 5,
                "tags": ["a", "b"],
                "status": "pending",
            },
            {
                "messages": ["world"],
                "counts": 3,
                "tags": ["b", "c"],
                "status": "done",
            },
        ]

        result = merger.merge(states)

        # Messages appended
        assert result["messages"] == ["hello", "world"]

        # Counts summed
        assert result["counts"] == 8

        # Tags unioned
        assert set(result["tags"]) == {"a", "b", "c"}

        # Status uses default (last_value)
        assert result["status"] == "done"

    def test_merge_checkpoints(self):
        """Test merging checkpoint states."""
        config = ReducerConfig(
            field_reducers={"events": "append"},
            default_reducer="last_value",
        )
        merger = StateMerger(config)

        checkpoints = [
            Checkpoint(state={"events": [1], "step": "a"}),
            Checkpoint(state={"events": [2], "step": "b"}),
        ]

        result = merger.merge_checkpoints(checkpoints)
        assert result["events"] == [1, 2]
        assert result["step"] == "b"


# =============================================================================
# merge_states Function Tests
# =============================================================================


class TestMergeStatesFunction:
    """Tests for merge_states convenience function."""

    def test_merge_states_basic(self):
        """Test basic merge_states usage."""
        states = [
            {"a": 1},
            {"b": 2},
        ]
        result = merge_states(states)
        assert result == {"a": 1, "b": 2}

    def test_merge_states_with_config(self):
        """Test merge_states with custom config."""
        config = ReducerConfig(
            field_reducers={"total": "sum"},
            default_reducer="last_value",
        )
        states = [
            {"total": 10, "name": "a"},
            {"total": 20, "name": "b"},
        ]
        result = merge_states(states, config)
        assert result["total"] == 30
        assert result["name"] == "b"

    def test_merge_states_empty(self):
        """Test merge_states with empty list."""
        result = merge_states([])
        assert result == {}


# =============================================================================
# Integration Tests
# =============================================================================


class TestReducerIntegration:
    """Integration tests for reducer system."""

    def test_complex_workflow_merge(self):
        """Test complex workflow state merge scenario."""
        config = ReducerConfig(
            field_reducers={
                "logs": "append",
                "errors": "append",
                "context": "merge_dict",
                "total_time_ms": "sum",
                "max_retries": "max",
                "min_confidence": "min",
                "visited_nodes": "union",
            },
            default_reducer="last_value",
        )
        merger = StateMerger(config)

        # Simulate 3 parallel branch states
        branch_states = [
            {
                "logs": ["Branch A started", "Branch A completed"],
                "errors": [],
                "context": {"branch": "A", "result": "success"},
                "total_time_ms": 100,
                "max_retries": 2,
                "min_confidence": 0.9,
                "visited_nodes": ["node1", "node2"],
                "final_status": "ok",
            },
            {
                "logs": ["Branch B started", "Branch B retry", "Branch B completed"],
                "errors": ["Retry needed on step 2"],
                "context": {"branch": "B", "result": "success"},
                "total_time_ms": 250,
                "max_retries": 5,
                "min_confidence": 0.7,
                "visited_nodes": ["node2", "node3"],
                "final_status": "ok",
            },
            {
                "logs": ["Branch C started", "Branch C completed"],
                "errors": [],
                "context": {"branch": "C", "result": "success"},
                "total_time_ms": 50,
                "max_retries": 0,
                "min_confidence": 0.95,
                "visited_nodes": ["node3", "node4"],
                "final_status": "ok",
            },
        ]

        result = merger.merge(branch_states)

        # All logs combined
        assert len(result["logs"]) == 7
        assert "Branch A started" in result["logs"]
        assert "Branch B retry" in result["logs"]

        # Errors combined
        assert len(result["errors"]) == 1
        assert "Retry needed" in result["errors"][0]

        # Context merged (later wins for overlapping keys)
        assert result["context"]["branch"] == "C"
        assert result["context"]["result"] == "success"

        # Times summed
        assert result["total_time_ms"] == 400  # 100 + 250 + 50

        # Max retries
        assert result["max_retries"] == 5

        # Min confidence
        assert result["min_confidence"] == 0.7

        # Visited nodes unioned
        assert set(result["visited_nodes"]) == {"node1", "node2", "node3", "node4"}

        # Final status (last value)
        assert result["final_status"] == "ok"
