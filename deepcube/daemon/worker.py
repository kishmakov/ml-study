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
from search.a_star import a_star_search
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
    encode_updated,
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
        result = a_star_search(
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


def run_worker(worker_endpoint: str) -> None:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    context = zmq.Context.instance()
    worker = context.socket(zmq.DEALER)
    worker.connect(worker_endpoint)
    worker_id = str(getpid())
    worker.send_string(encode_ready(worker_id))

    poller = zmq.Poller()
    poller.register(worker, zmq.POLLIN)

    state = WorkerState()
    try:
        while True:
            events = dict(poller.poll())
            if worker in events:
                raw = worker.recv_string()
                command = raw.split("\t", 1)[0]
                if command == STOP:
                    break
                if command == UPDATE:
                    update = decode_update(raw)
                    state.load(
                        update.puzzle_name,
                        update.model_stem,
                        update.model_version,
                    )
                    worker.send_string(encode_updated(worker_id, update.model_version))
                elif command == TASK:
                    task = decode_task(raw)
                    try:
                        worker.send_string(encode_result(state.evaluate(task)))
                    except Exception:
                        worker.send_string(
                            encode_error(
                                ErrorMessage(task.client_id, task.task_id, format_exc())
                            )
                        )
                else:
                    raise AssertionError(f"invalid worker command: {raw!r}")
    except KeyboardInterrupt:
        pass

    worker.close(linger=0)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--worker-endpoint", required=True)
    args = parser.parse_args()
    run_worker(args.worker_endpoint)


if __name__ == "__main__":
    main()
