"""Client-side adapter for the ZeroMQ CTG daemon."""

from __future__ import annotations

from os.path import join
from shutil import rmtree
from subprocess import DEVNULL, Popen, TimeoutExpired
from sys import executable
from tempfile import mkdtemp
from time import sleep
from uuid import uuid4

from tqdm import tqdm
import zmq

from puzzle import StateKey

from daemon.protocol import (
    ERROR,
    READY,
    RESULT,
    TaskMessage,
    UpdateMessage,
    decode_error,
    decode_result,
    encode_task,
    encode_update,
)


DEFAULT_TIMEOUT_MS = 30 * 60 * 1000
DEFAULT_WORKERS = 12
STARTUP_WAIT_SEC = 0.75


class CtgZmqClient:
    """Submit CTG tasks to local ZMQ workers and collect replies."""

    def __init__(
        self,
        task_endpoint: str,
        update_endpoint: str,
        result_endpoint: str,
        timeout_ms: int,
        managed_workers: list[Popen[str]],
        endpoint_dir: str | None,
    ) -> None:
        self.client_id = uuid4().hex
        self.timeout_ms = timeout_ms
        self.managed_workers = managed_workers
        self.endpoint_dir = endpoint_dir
        self.context = zmq.Context.instance()

        self.tasks = self.context.socket(zmq.PUSH)
        self.tasks.bind(task_endpoint)

        self.updates = self.context.socket(zmq.PUB)
        self.updates.bind(update_endpoint)

        self.results = self.context.socket(zmq.PULL)
        self.results.bind(result_endpoint)

    @classmethod
    def start(cls) -> "CtgZmqClient":
        endpoint_dir = mkdtemp(prefix="deepcube-ctg-")
        task_endpoint = _ipc_endpoint(endpoint_dir, "tasks.sock")
        update_endpoint = _ipc_endpoint(endpoint_dir, "updates.sock")
        result_endpoint = _ipc_endpoint(endpoint_dir, "results.sock")
        client = cls(
            task_endpoint,
            update_endpoint,
            result_endpoint,
            DEFAULT_TIMEOUT_MS,
            [],
            endpoint_dir,
        )
        client.managed_workers = start_local_workers(
            DEFAULT_WORKERS,
            task_endpoint,
            update_endpoint,
            result_endpoint,
        )
        client.wait_for_workers_ready(DEFAULT_WORKERS)
        return client

    def wait_for_workers_ready(self, workers: int) -> None:
        ready = 0
        poller = zmq.Poller()
        poller.register(self.results, zmq.POLLIN)

        while ready < workers:
            assert all(process.poll() is None for process in self.managed_workers), (
                "ZMQ CTG worker exited during startup"
            )
            events = dict(poller.poll(DEFAULT_TIMEOUT_MS))
            assert self.results in events, (
                f"timed out waiting for {workers - ready} CTG worker(s)"
            )
            raw = self.results.recv_string()
            command = raw.split("\t", 1)[0]
            assert command == READY, f"unexpected startup message: {raw!r}"
            ready += 1
        sleep(STARTUP_WAIT_SEC)

    def evaluate(
        self,
        puzzle_name: str,
        model_stem: str,
        model_version: int,
        bunches: dict[int | str, list[StateKey]],
        progress_desc: str,
    ) -> tuple[dict[int, list[float]], dict[int, list[int]]]:
        tasks = [
            (int(depth), state)
            for depth, states in bunches.items()
            for state in states
        ]
        if not tasks:
            return {}, {}

        update = UpdateMessage(puzzle_name, model_stem, model_version)
        self.updates.send_string(encode_update(update))

        pending: set[str] = set()
        for index, (depth, state) in enumerate(tasks):
            task_id = str(index)
            pending.add(task_id)
            self.tasks.send_string(
                encode_task(
                    TaskMessage(
                        self.client_id,
                        task_id,
                        puzzle_name,
                        model_stem,
                        model_version,
                        depth,
                        state,
                    )
                )
            )

        values_by_depth: dict[int, list[float]] = {}
        solved_by_depth: dict[int, list[int]] = {}
        poller = zmq.Poller()
        poller.register(self.results, zmq.POLLIN)

        with tqdm(total=len(tasks), desc=progress_desc, leave=False) as progress_bar:
            while pending:
                events = dict(poller.poll(self.timeout_ms))
                assert self.results in events, (
                    f"timed out waiting for {len(pending)} ZMQ CTG result(s)"
                )
                raw = self.results.recv_string()
                command = raw.split("\t", 1)[0]
                if command == RESULT:
                    result = decode_result(raw)
                    if result.task_id not in pending:
                        continue
                    pending.remove(result.task_id)
                    values_by_depth.setdefault(result.depth, []).append(result.value)
                    solved_by_depth.setdefault(result.depth, []).append(result.solved)
                    progress_bar.update(1)
                elif command == ERROR:
                    error = decode_error(raw)
                    if error.task_id in pending:
                        raise RuntimeError(f"ZMQ CTG worker failed: {error.message}")
                elif command == READY:
                    continue

        return values_by_depth, solved_by_depth

    def close(self) -> None:
        self.tasks.close(linger=0)
        self.updates.close(linger=0)
        self.results.close(linger=0)
        stop_local_workers(self.managed_workers)
        self.managed_workers = []
        if self.endpoint_dir is not None:
            rmtree(self.endpoint_dir, ignore_errors=True)
            self.endpoint_dir = None


def start_local_workers(
    workers: int,
    task_endpoint: str,
    update_endpoint: str,
    result_endpoint: str,
) -> list[Popen[str]]:
    processes: list[Popen[str]] = []
    for _ in range(workers):
        processes.append(
            Popen(
                [
                    executable,
                    "-m",
                    "daemon.worker",
                    "--task-endpoint",
                    task_endpoint,
                    "--update-endpoint",
                    update_endpoint,
                    "--result-endpoint",
                    result_endpoint,
                ],
                stdin=DEVNULL,
                stdout=DEVNULL,
                stderr=DEVNULL,
                text=True,
            )
        )
    return processes


def stop_local_workers(processes: list[Popen[str]]) -> None:
    for process in processes:
        process.terminate()
    for process in processes:
        try:
            process.wait(timeout=5)
        except TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def _ipc_endpoint(endpoint_dir: str, name: str) -> str:
    return "ipc://" + join(endpoint_dir, name)
