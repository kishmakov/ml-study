"""Python CTG worker process for the ZeroMQ daemon executor.

The wire protocol is intentionally string-based so a Mojo worker can replace
this module without changing the Python caller.
"""

from __future__ import annotations

from argparse import ArgumentParser
from os import getpid
from traceback import format_exc

import torch
import zmq

from costtogo import NeuralCostToGo
from puzzle import Puzzle
from puzzle_factory import create_puzzle
from search_a_star import solve_a_star
from train_costtogo import (
    CTG_EVAL_MAX_STATES,
    CTG_EVAL_POP_BATCH_SIZE,
    CTG_EVAL_WEIGHT,
)

from daemon.protocol import (
    STOP,
    TASK,
    UPDATE,
    ErrorMessage,
    ResultMessage,
    TaskMessage,
    decode_task,
    decode_update,
    encode_error,
    encode_ready,
    encode_result,
)


class WorkerState:
    def __init__(self) -> None:
        self.puzzle_name: str | None = None
        self.model_stem: str | None = None
        self.model_version: int | None = None
        self.puzzle: Puzzle | None = None
        self.heuristic: NeuralCostToGo | None = None

    def load(self, puzzle_name: str, model_stem: str, model_version: int) -> None:
        if (
            self.puzzle is not None
            and self.heuristic is not None
            and self.puzzle_name == puzzle_name
            and self.model_stem == model_stem
            and self.model_version == model_version
        ):
            return

        puzzle = create_puzzle(puzzle_name)
        heuristic = NeuralCostToGo.from_checkpoint(model_stem + ".pt", puzzle, "cpu")
        self.puzzle_name = puzzle_name
        self.model_stem = model_stem
        self.model_version = model_version
        self.puzzle = puzzle
        self.heuristic = heuristic

    def evaluate(self, task: TaskMessage) -> ResultMessage:
        self.load(task.puzzle_name, task.model_stem, task.model_version)
        assert self.puzzle is not None, "worker puzzle was not loaded"
        assert self.heuristic is not None, "worker heuristic was not loaded"

        self.puzzle.reset(task.state)
        value = self.heuristic(self.puzzle.cost_to_go_input())

        self.puzzle.reset(task.state)
        result = solve_a_star(
            self.puzzle,
            self.heuristic.batch,
            CTG_EVAL_WEIGHT,
            CTG_EVAL_MAX_STATES,
            CTG_EVAL_POP_BATCH_SIZE,
        )
        return ResultMessage(
            task.client_id,
            task.task_id,
            task.depth,
            value,
            int(result.solved),
        )


def run_worker(task_endpoint: str, update_endpoint: str, result_endpoint: str) -> None:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    context = zmq.Context.instance()
    tasks = context.socket(zmq.PULL)
    tasks.connect(task_endpoint)

    updates = context.socket(zmq.SUB)
    updates.setsockopt_string(zmq.SUBSCRIBE, "")
    updates.connect(update_endpoint)

    results = context.socket(zmq.PUSH)
    results.connect(result_endpoint)
    results.send_string(encode_ready(str(getpid())))

    poller = zmq.Poller()
    poller.register(tasks, zmq.POLLIN)
    poller.register(updates, zmq.POLLIN)

    state = WorkerState()
    try:
        while True:
            events = dict(poller.poll())
            if updates in events:
                raw = updates.recv_string()
                if raw == STOP:
                    break
                if raw.startswith(UPDATE + "\t"):
                    update = decode_update(raw)
                    state.load(
                        update.puzzle_name,
                        update.model_stem,
                        update.model_version,
                    )

            if tasks in events:
                raw = tasks.recv_string()
                if raw == STOP:
                    break
                assert raw.startswith(TASK + "\t"), f"invalid worker task: {raw!r}"
                task = decode_task(raw)
                try:
                    results.send_string(encode_result(state.evaluate(task)))
                except Exception:
                    results.send_string(
                        encode_error(
                            ErrorMessage(task.client_id, task.task_id, format_exc())
                        )
                    )
    except KeyboardInterrupt:
        pass

    tasks.close(linger=0)
    updates.close(linger=0)
    results.close(linger=0)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--task-endpoint", required=True)
    parser.add_argument("--update-endpoint", required=True)
    parser.add_argument("--result-endpoint", required=True)
    args = parser.parse_args()
    run_worker(args.task_endpoint, args.update_endpoint, args.result_endpoint)


if __name__ == "__main__":
    main()
