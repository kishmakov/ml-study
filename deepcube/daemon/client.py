"""Client-side adapter for the ZeroMQ CTG daemon."""

from __future__ import annotations

from os.path import dirname, exists, join
from shutil import rmtree
from subprocess import DEVNULL, Popen, TimeoutExpired
from tempfile import mkdtemp
from uuid import uuid4

from tqdm import tqdm
import zmq

from puzzle.puzzle import StateKey
from puzzle_factory import MODEL_DIR

from daemon.protocol import (
    ERROR,
    READY,
    RESULT,
    UPDATED,
    TaskMessage,
    UpdateMessage,
    decode_error,
    decode_result,
    encode_task,
    encode_update,
)


DEFAULT_TIMEOUT_MS = 30 * 60 * 1000
DEFAULT_WORKERS = 15
NATIVE_WORKER = join(dirname(__file__), "worker")


class CtgZmqClient:
    """Submit CTG tasks to local ZMQ workers and collect replies."""

    def __init__(
        self,
        worker_endpoint: str,
        timeout_ms: int,
        managed_workers: list[Popen[str]],
        endpoint_dir: str | None,
    ) -> None:
        self.client_id = uuid4().hex
        self.timeout_ms = timeout_ms
        self.managed_workers = managed_workers
        self.endpoint_dir = endpoint_dir
        self.worker_ids: list[bytes] = []
        self.context = zmq.Context.instance()

        self.workers = self.context.socket(zmq.ROUTER)
        self.workers.bind(worker_endpoint)

    @classmethod
    def start(cls, max_states: int) -> "CtgZmqClient":
        worker_endpoint = _ipc_endpoint(MODEL_DIR, "workers.sock")
        client = cls(
            worker_endpoint,
            DEFAULT_TIMEOUT_MS,
            [],
            MODEL_DIR,
        )
        client.managed_workers = start_local_workers(
            DEFAULT_WORKERS,
            worker_endpoint,
            max_states,
        )
        client.wait_for_workers_ready(DEFAULT_WORKERS)
        return client

    def wait_for_workers_ready(self, workers: int) -> None:
        poller = zmq.Poller()
        poller.register(self.workers, zmq.POLLIN)

        while len(self.worker_ids) < workers:
            assert all(process.poll() is None for process in self.managed_workers), (
                "ZMQ CTG worker exited during startup"
            )
            events = dict(poller.poll(DEFAULT_TIMEOUT_MS))
            assert self.workers in events, (
                f"timed out waiting for {workers - len(self.worker_ids)} CTG worker(s)"
            )
            worker_id, raw = self.receive_worker_message()
            command = raw.split("\t", 1)[0]
            assert command == READY, f"unexpected startup message: {raw!r}"
            if worker_id not in self.worker_ids:
                self.worker_ids.append(worker_id)

    def wait_for_model_update(self, update: UpdateMessage) -> None:
        updated: set[bytes] = set()
        poller = zmq.Poller()
        poller.register(self.workers, zmq.POLLIN)
        update_raw = encode_update(update)
        for worker_id in self.worker_ids:
            self.send_worker_message(worker_id, update_raw)

        while len(updated) < len(self.worker_ids):
            assert all(process.poll() is None for process in self.managed_workers), (
                "ZMQ CTG worker exited while loading model"
            )
            events = dict(poller.poll(250))
            if self.workers not in events:
                for worker_id in self.worker_ids:
                    if worker_id not in updated:
                        self.send_worker_message(worker_id, update_raw)
                continue

            worker_id, raw = self.receive_worker_message()
            command = raw.split("\t", 1)[0]
            if command == UPDATED:
                parts = raw.split("\t")
                assert len(parts) == 3, f"invalid model update ack: {raw!r}"
                if int(parts[2]) == update.model_version:
                    updated.add(worker_id)
            elif command == READY:
                continue
            else:
                raise AssertionError(f"unexpected model update message: {raw!r}")

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
        self.wait_for_model_update(update)

        task_iter = iter(enumerate(reversed(tasks)))
        pending: set[str] = set()
        idle_workers = list(self.worker_ids)
        values_by_depth: dict[int, list[float]] = {}
        solved_by_depth: dict[int, list[int]] = {}
        poller = zmq.Poller()
        poller.register(self.workers, zmq.POLLIN)

        def send_next_task(worker_id: bytes) -> bool:
            try:
                index, (depth, state) = next(task_iter)
            except StopIteration:
                return False

            task_id = str(index)
            pending.add(task_id)
            self.send_worker_message(
                worker_id,
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
            return True

        while idle_workers and send_next_task(idle_workers.pop()):
            pass

        with tqdm(total=len(tasks), desc=progress_desc, leave=False) as progress_bar:
            while pending:
                events = dict(poller.poll(self.timeout_ms))
                assert self.workers in events, (
                    f"timed out waiting for {len(pending)} ZMQ CTG result(s)"
                )
                worker_id, raw = self.receive_worker_message()
                command = raw.split("\t", 1)[0]
                if command == RESULT:
                    result = decode_result(raw)
                    if result.task_id not in pending:
                        continue
                    pending.remove(result.task_id)
                    values_by_depth.setdefault(result.depth, []).append(result.value)
                    solved_by_depth.setdefault(result.depth, []).append(result.solved)
                    progress_bar.update(1)
                    send_next_task(worker_id)
                elif command == ERROR:
                    error = decode_error(raw)
                    if error.task_id in pending:
                        raise RuntimeError(f"ZMQ CTG worker failed: {error.message}")
                elif command == READY:
                    continue
                elif command == UPDATED:
                    continue

        return values_by_depth, solved_by_depth

    def receive_worker_message(self) -> tuple[bytes, str]:
        frames = self.workers.recv_multipart()
        assert len(frames) == 2, f"invalid worker message frames: {frames!r}"
        return frames[0], frames[1].decode("utf-8")

    def send_worker_message(self, worker_id: bytes, message: str) -> None:
        self.workers.send_multipart([worker_id, message.encode("utf-8")])

    def close(self) -> None:
        self.workers.close(linger=0)
        stop_local_workers(self.managed_workers)
        self.managed_workers = []
        self.endpoint_dir = None


def start_local_workers(
    workers: int,
    worker_endpoint: str,
    max_states: int,
) -> list[Popen[str]]:
    assert exists(NATIVE_WORKER), "native worker is missing; run ./build_native.sh"

    processes: list[Popen[str]] = []
    for _ in range(workers):
        processes.append(
            Popen(
                [
                    NATIVE_WORKER,
                    "--worker-endpoint",
                    worker_endpoint,
                    "--max-states",
                    str(max_states),
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
