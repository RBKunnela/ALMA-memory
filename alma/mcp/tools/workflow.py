"""
ALMA MCP Workflow Tools.

Provides sync and async workflow tool functions for the MCP protocol.
Includes checkpointing, resume, state merging, workflow learning,
artifact linking, and checkpoint cleanup.
"""

import logging
from typing import Any, Dict, Optional

from alma import ALMA

logger = logging.getLogger(__name__)


def alma_checkpoint(
    alma: ALMA,
    run_id: str,
    node_id: str,
    state: Dict[str, Any],
    branch_id: Optional[str] = None,
    parent_checkpoint_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    skip_if_unchanged: bool = True,
) -> Dict[str, Any]:
    """
    Create a checkpoint for crash recovery.

    Args:
        alma: ALMA instance
        run_id: The workflow run identifier
        node_id: The node creating this checkpoint
        state: The state to persist
        branch_id: Optional branch identifier for parallel execution
        parent_checkpoint_id: Previous checkpoint in the chain
        metadata: Additional checkpoint metadata
        skip_if_unchanged: If True, skip if state hasn't changed

    Returns:
        Dict with checkpoint info or skip notification
    """
    if not run_id or not run_id.strip():
        return {"success": False, "error": "run_id cannot be empty"}
    if not node_id or not node_id.strip():
        return {"success": False, "error": "node_id cannot be empty"}

    try:
        checkpoint = alma.checkpoint(
            run_id=run_id,
            node_id=node_id,
            state=state,
            branch_id=branch_id,
            parent_checkpoint_id=parent_checkpoint_id,
            metadata=metadata,
            skip_if_unchanged=skip_if_unchanged,
        )

        if checkpoint is None:
            return {
                "success": True,
                "skipped": True,
                "message": "Checkpoint skipped - state unchanged",
            }

        return {
            "success": True,
            "checkpoint": {
                "id": checkpoint.id,
                "run_id": checkpoint.run_id,
                "node_id": checkpoint.node_id,
                "sequence_number": checkpoint.sequence_number,
                "branch_id": checkpoint.branch_id,
                "state_hash": checkpoint.state_hash,
                "created_at": checkpoint.created_at.isoformat(),
            },
        }

    except Exception as e:
        logger.exception(f"Error in alma_checkpoint: {e}")
        return {"success": False, "error": str(e)}


def alma_resume(
    alma: ALMA,
    run_id: str,
    branch_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get the checkpoint to resume from after a crash.

    Args:
        alma: ALMA instance
        run_id: The workflow run identifier
        branch_id: Optional branch to filter by

    Returns:
        Dict with checkpoint info or None if no checkpoints
    """
    if not run_id or not run_id.strip():
        return {"success": False, "error": "run_id cannot be empty"}

    try:
        checkpoint = alma.get_resume_point(run_id, branch_id)

        if checkpoint is None:
            return {
                "success": True,
                "checkpoint": None,
                "message": "No checkpoint found for this run",
            }

        return {
            "success": True,
            "checkpoint": {
                "id": checkpoint.id,
                "run_id": checkpoint.run_id,
                "node_id": checkpoint.node_id,
                "state": checkpoint.state,
                "sequence_number": checkpoint.sequence_number,
                "branch_id": checkpoint.branch_id,
                "parent_checkpoint_id": checkpoint.parent_checkpoint_id,
                "created_at": checkpoint.created_at.isoformat(),
            },
        }

    except Exception as e:
        logger.exception(f"Error in alma_resume: {e}")
        return {"success": False, "error": str(e)}


def alma_merge_states(
    alma: ALMA,
    states: list,
    reducer_config: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Merge multiple branch states after parallel execution.

    Args:
        alma: ALMA instance
        states: List of state dicts from parallel branches
        reducer_config: Optional mapping of key -> reducer name.
                       Available: append, merge_dict, last_value,
                       first_value, sum, max, min, union

    Returns:
        Dict with merged state
    """
    if not states:
        return {"success": True, "merged_state": {}}

    try:
        merged = alma.merge_states(states, reducer_config)

        return {
            "success": True,
            "merged_state": merged,
            "input_count": len(states),
        }

    except Exception as e:
        logger.exception(f"Error in alma_merge_states: {e}")
        return {"success": False, "error": str(e)}


def alma_workflow_learn(
    alma: ALMA,
    agent: str,
    workflow_id: str,
    run_id: str,
    result: str,
    summary: str,
    strategies_used: Optional[list] = None,
    successful_patterns: Optional[list] = None,
    failed_patterns: Optional[list] = None,
    duration_seconds: Optional[float] = None,
    node_count: Optional[int] = None,
    error_message: Optional[str] = None,
    tenant_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Record learnings from a completed workflow execution.

    Args:
        alma: ALMA instance
        agent: The agent that executed the workflow
        workflow_id: The workflow definition that was executed
        run_id: The specific run this outcome is from
        result: Result status (success, failure, partial, cancelled, timeout)
        summary: Human-readable summary of what happened
        strategies_used: List of strategies/approaches attempted
        successful_patterns: Patterns that worked well
        failed_patterns: Patterns that didn't work
        duration_seconds: How long the workflow took
        node_count: Number of nodes executed
        error_message: Error details if failed
        tenant_id: Multi-tenant isolation identifier
        metadata: Additional outcome metadata

    Returns:
        Dict with workflow outcome info
    """
    if not agent or not agent.strip():
        return {"success": False, "error": "agent cannot be empty"}
    if not workflow_id or not workflow_id.strip():
        return {"success": False, "error": "workflow_id cannot be empty"}
    if not run_id or not run_id.strip():
        return {"success": False, "error": "run_id cannot be empty"}
    if not result or not result.strip():
        return {"success": False, "error": "result cannot be empty"}
    if not summary or not summary.strip():
        return {"success": False, "error": "summary cannot be empty"}

    valid_results = ["success", "failure", "partial", "cancelled", "timeout"]
    if result not in valid_results:
        return {
            "success": False,
            "error": f"result must be one of: {', '.join(valid_results)}",
        }

    try:
        outcome = alma.learn_from_workflow(
            agent=agent,
            workflow_id=workflow_id,
            run_id=run_id,
            result=result,
            summary=summary,
            strategies_used=strategies_used,
            successful_patterns=successful_patterns,
            failed_patterns=failed_patterns,
            duration_seconds=duration_seconds,
            node_count=node_count,
            error_message=error_message,
            tenant_id=tenant_id,
            metadata=metadata,
        )

        return {
            "success": True,
            "outcome": {
                "id": outcome.id,
                "workflow_id": outcome.workflow_id,
                "run_id": outcome.run_id,
                "result": outcome.result.value,
                "agent": outcome.agent,
                "created_at": outcome.created_at.isoformat(),
            },
            "message": "Workflow outcome recorded successfully",
        }

    except Exception as e:
        logger.exception(f"Error in alma_workflow_learn: {e}")
        return {"success": False, "error": str(e)}


def alma_link_artifact(
    alma: ALMA,
    memory_id: str,
    artifact_type: str,
    storage_url: str,
    filename: Optional[str] = None,
    mime_type: Optional[str] = None,
    size_bytes: Optional[int] = None,
    checksum: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Link an external artifact to a memory.

    Args:
        alma: ALMA instance
        memory_id: The memory to link the artifact to
        artifact_type: Type (screenshot, log, report, file, etc.)
        storage_url: URL or path to the artifact in storage
        filename: Original filename
        mime_type: MIME type
        size_bytes: Size in bytes
        checksum: SHA256 checksum for integrity
        metadata: Additional artifact metadata

    Returns:
        Dict with artifact reference info
    """
    if not memory_id or not memory_id.strip():
        return {"success": False, "error": "memory_id cannot be empty"}
    if not artifact_type or not artifact_type.strip():
        return {"success": False, "error": "artifact_type cannot be empty"}
    if not storage_url or not storage_url.strip():
        return {"success": False, "error": "storage_url cannot be empty"}

    try:
        artifact = alma.link_artifact(
            memory_id=memory_id,
            artifact_type=artifact_type,
            storage_url=storage_url,
            filename=filename,
            mime_type=mime_type,
            size_bytes=size_bytes,
            checksum=checksum,
            metadata=metadata,
        )

        return {
            "success": True,
            "artifact": {
                "id": artifact.id,
                "memory_id": artifact.memory_id,
                "artifact_type": artifact.artifact_type.value,
                "storage_url": artifact.storage_url,
                "filename": artifact.filename,
                "created_at": artifact.created_at.isoformat(),
            },
        }

    except Exception as e:
        logger.exception(f"Error in alma_link_artifact: {e}")
        return {"success": False, "error": str(e)}


def alma_get_artifacts(
    alma: ALMA,
    memory_id: str,
) -> Dict[str, Any]:
    """
    Get all artifacts linked to a memory.

    Args:
        alma: ALMA instance
        memory_id: The memory to get artifacts for

    Returns:
        Dict with list of artifact references
    """
    if not memory_id or not memory_id.strip():
        return {"success": False, "error": "memory_id cannot be empty"}

    try:
        artifacts = alma.get_artifacts(memory_id)

        return {
            "success": True,
            "artifacts": [
                {
                    "id": a.id,
                    "artifact_type": a.artifact_type.value,
                    "storage_url": a.storage_url,
                    "filename": a.filename,
                    "mime_type": a.mime_type,
                    "size_bytes": a.size_bytes,
                    "created_at": a.created_at.isoformat(),
                }
                for a in artifacts
            ],
            "count": len(artifacts),
        }

    except Exception as e:
        logger.exception(f"Error in alma_get_artifacts: {e}")
        return {"success": False, "error": str(e)}


def alma_cleanup_checkpoints(
    alma: ALMA,
    run_id: str,
    keep_latest: int = 1,
) -> Dict[str, Any]:
    """
    Clean up old checkpoints for a completed run.

    Args:
        alma: ALMA instance
        run_id: The workflow run identifier
        keep_latest: Number of latest checkpoints to keep

    Returns:
        Dict with cleanup results
    """
    if not run_id or not run_id.strip():
        return {"success": False, "error": "run_id cannot be empty"}

    try:
        count = alma.cleanup_checkpoints(run_id, keep_latest)

        return {
            "success": True,
            "deleted_count": count,
            "kept": keep_latest,
            "message": f"Deleted {count} checkpoints, kept {keep_latest}",
        }

    except Exception as e:
        logger.exception(f"Error in alma_cleanup_checkpoints: {e}")
        return {"success": False, "error": str(e)}


# =============================================================================
# ASYNC WORKFLOW TOOLS
# =============================================================================


async def async_alma_checkpoint(
    alma: ALMA,
    run_id: str,
    node_id: str,
    state: Dict[str, Any],
    branch_id: Optional[str] = None,
    parent_checkpoint_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    skip_if_unchanged: bool = True,
) -> Dict[str, Any]:
    """Async version of alma_checkpoint."""
    if not run_id or not run_id.strip():
        return {"success": False, "error": "run_id cannot be empty"}
    if not node_id or not node_id.strip():
        return {"success": False, "error": "node_id cannot be empty"}

    try:
        checkpoint = await alma.async_checkpoint(
            run_id=run_id,
            node_id=node_id,
            state=state,
            branch_id=branch_id,
            parent_checkpoint_id=parent_checkpoint_id,
            metadata=metadata,
            skip_if_unchanged=skip_if_unchanged,
        )

        if checkpoint is None:
            return {
                "success": True,
                "skipped": True,
                "message": "Checkpoint skipped - state unchanged",
            }

        return {
            "success": True,
            "checkpoint": {
                "id": checkpoint.id,
                "run_id": checkpoint.run_id,
                "node_id": checkpoint.node_id,
                "sequence_number": checkpoint.sequence_number,
                "branch_id": checkpoint.branch_id,
                "state_hash": checkpoint.state_hash,
                "created_at": checkpoint.created_at.isoformat(),
            },
        }

    except Exception as e:
        logger.exception(f"Error in async_alma_checkpoint: {e}")
        return {"success": False, "error": str(e)}


async def async_alma_resume(
    alma: ALMA,
    run_id: str,
    branch_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Async version of alma_resume."""
    if not run_id or not run_id.strip():
        return {"success": False, "error": "run_id cannot be empty"}

    try:
        checkpoint = await alma.async_get_resume_point(run_id, branch_id)

        if checkpoint is None:
            return {
                "success": True,
                "checkpoint": None,
                "message": "No checkpoint found for this run",
            }

        return {
            "success": True,
            "checkpoint": {
                "id": checkpoint.id,
                "run_id": checkpoint.run_id,
                "node_id": checkpoint.node_id,
                "state": checkpoint.state,
                "sequence_number": checkpoint.sequence_number,
                "branch_id": checkpoint.branch_id,
                "parent_checkpoint_id": checkpoint.parent_checkpoint_id,
                "created_at": checkpoint.created_at.isoformat(),
            },
        }

    except Exception as e:
        logger.exception(f"Error in async_alma_resume: {e}")
        return {"success": False, "error": str(e)}


async def async_alma_workflow_learn(
    alma: ALMA,
    agent: str,
    workflow_id: str,
    run_id: str,
    result: str,
    summary: str,
    strategies_used: Optional[list] = None,
    successful_patterns: Optional[list] = None,
    failed_patterns: Optional[list] = None,
    duration_seconds: Optional[float] = None,
    node_count: Optional[int] = None,
    error_message: Optional[str] = None,
    tenant_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Async version of alma_workflow_learn."""
    if not agent or not agent.strip():
        return {"success": False, "error": "agent cannot be empty"}
    if not workflow_id or not workflow_id.strip():
        return {"success": False, "error": "workflow_id cannot be empty"}
    if not run_id or not run_id.strip():
        return {"success": False, "error": "run_id cannot be empty"}
    if not result or not result.strip():
        return {"success": False, "error": "result cannot be empty"}
    if not summary or not summary.strip():
        return {"success": False, "error": "summary cannot be empty"}

    valid_results = ["success", "failure", "partial", "cancelled", "timeout"]
    if result not in valid_results:
        return {
            "success": False,
            "error": f"result must be one of: {', '.join(valid_results)}",
        }

    try:
        outcome = await alma.async_learn_from_workflow(
            agent=agent,
            workflow_id=workflow_id,
            run_id=run_id,
            result=result,
            summary=summary,
            strategies_used=strategies_used,
            successful_patterns=successful_patterns,
            failed_patterns=failed_patterns,
            duration_seconds=duration_seconds,
            node_count=node_count,
            error_message=error_message,
            tenant_id=tenant_id,
            metadata=metadata,
        )

        return {
            "success": True,
            "outcome": {
                "id": outcome.id,
                "workflow_id": outcome.workflow_id,
                "run_id": outcome.run_id,
                "result": outcome.result.value,
                "agent": outcome.agent,
                "created_at": outcome.created_at.isoformat(),
            },
            "message": "Workflow outcome recorded successfully",
        }

    except Exception as e:
        logger.exception(f"Error in async_alma_workflow_learn: {e}")
        return {"success": False, "error": str(e)}


async def async_alma_link_artifact(
    alma: ALMA,
    memory_id: str,
    artifact_type: str,
    storage_url: str,
    filename: Optional[str] = None,
    mime_type: Optional[str] = None,
    size_bytes: Optional[int] = None,
    checksum: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Async version of alma_link_artifact."""
    if not memory_id or not memory_id.strip():
        return {"success": False, "error": "memory_id cannot be empty"}
    if not artifact_type or not artifact_type.strip():
        return {"success": False, "error": "artifact_type cannot be empty"}
    if not storage_url or not storage_url.strip():
        return {"success": False, "error": "storage_url cannot be empty"}

    try:
        artifact = await alma.async_link_artifact(
            memory_id=memory_id,
            artifact_type=artifact_type,
            storage_url=storage_url,
            filename=filename,
            mime_type=mime_type,
            size_bytes=size_bytes,
            checksum=checksum,
            metadata=metadata,
        )

        return {
            "success": True,
            "artifact": {
                "id": artifact.id,
                "memory_id": artifact.memory_id,
                "artifact_type": artifact.artifact_type.value,
                "storage_url": artifact.storage_url,
                "filename": artifact.filename,
                "created_at": artifact.created_at.isoformat(),
            },
        }

    except Exception as e:
        logger.exception(f"Error in async_alma_link_artifact: {e}")
        return {"success": False, "error": str(e)}
