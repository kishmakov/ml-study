"""String protocol shared by the ZMQ CTG client and workers."""

from __future__ import annotations

from dataclasses import dataclass


TASK = "TASK"
UPDATE = "UPDATE"
UPDATED = "UPDATED"
RESULT = "RESULT"
ERROR = "ERROR"
READY = "READY"
STOP = "STOP"


@dataclass(frozen=True, slots=True)
class TaskMessage:
    client_id: str
    task_id: str
    puzzle_name: str
    model_stem: str
    model_version: int
    depth: int
    state: str


@dataclass(frozen=True, slots=True)
class UpdateMessage:
    puzzle_name: str
    model_stem: str
    model_version: int


@dataclass(frozen=True, slots=True)
class ResultMessage:
    client_id: str
    task_id: str
    depth: int
    value: float
    solved: int


@dataclass(frozen=True, slots=True)
class ErrorMessage:
    client_id: str
    task_id: str
    message: str


def encode_task(message: TaskMessage) -> str:
    return "\t".join(
        (
            TASK,
            message.client_id,
            message.task_id,
            message.puzzle_name,
            message.model_stem,
            str(message.model_version),
            str(message.depth),
            message.state,
        )
    )


def decode_task(raw: str) -> TaskMessage:
    parts = raw.split("\t", 7)
    assert len(parts) == 8 and parts[0] == TASK, f"invalid task message: {raw!r}"
    return TaskMessage(
        parts[1],
        parts[2],
        parts[3],
        parts[4],
        int(parts[5]),
        int(parts[6]),
        parts[7],
    )


def encode_update(message: UpdateMessage) -> str:
    return "\t".join(
        (
            UPDATE,
            message.puzzle_name,
            message.model_stem,
            str(message.model_version),
        )
    )


def decode_update(raw: str) -> UpdateMessage:
    parts = raw.split("\t")
    assert len(parts) == 4 and parts[0] == UPDATE, f"invalid update message: {raw!r}"
    return UpdateMessage(parts[1], parts[2], int(parts[3]))


def encode_result(message: ResultMessage) -> str:
    return "\t".join(
        (
            RESULT,
            message.client_id,
            message.task_id,
            str(message.depth),
            repr(message.value),
            str(message.solved),
        )
    )


def decode_result(raw: str) -> ResultMessage:
    parts = raw.split("\t")
    assert len(parts) == 6 and parts[0] == RESULT, f"invalid result message: {raw!r}"
    return ResultMessage(
        parts[1],
        parts[2],
        int(parts[3]),
        float(parts[4]),
        int(parts[5]),
    )


def encode_error(message: ErrorMessage) -> str:
    return "\t".join((ERROR, message.client_id, message.task_id, message.message))


def decode_error(raw: str) -> ErrorMessage:
    parts = raw.split("\t", 3)
    assert len(parts) == 4 and parts[0] == ERROR, f"invalid error message: {raw!r}"
    return ErrorMessage(parts[1], parts[2], parts[3])


def encode_ready(worker_id: str) -> str:
    return "\t".join((READY, worker_id))


def encode_updated(worker_id: str, model_version: int) -> str:
    return "\t".join((UPDATED, worker_id, str(model_version)))
