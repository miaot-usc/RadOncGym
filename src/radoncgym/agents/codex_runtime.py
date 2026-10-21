

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import json
import os
from pathlib import Path
import socket
import time
from typing import Iterator
import uuid

from radoncgym.core.errors import ContractError


@dataclass(frozen=True, slots=True)
class CodexGateLease:
    enabled: bool
    slot: int | None
    limit: int | None
    wait_seconds: float


@contextmanager
def _provider_concurrency_slot(
    *, provider_name: str, environment_prefix: str
) -> Iterator[CodexGateLease]:
    pass

    limit_name = f"{environment_prefix}_MAX_CONCURRENT"
    root_name = f"{environment_prefix}_GATE_ROOT"
    wait_name = f"{environment_prefix}_GATE_WAIT_SECONDS"
    raw_limit = os.environ.get(limit_name, "").strip()
    if not raw_limit:
        yield CodexGateLease(
            enabled=False, slot=None, limit=None, wait_seconds=0.0
        )
        return
    try:
        limit = int(raw_limit)
    except ValueError as error:
        raise ContractError(
            f"{limit_name} must be an integer"
        ) from error
    if limit < 1 or limit > 256:
        raise ContractError(
            f"{limit_name} must be between 1 and 256"
        )
    gate_root_raw = os.environ.get(root_name, "").strip()
    if not gate_root_raw:
        raise ContractError(
            f"{root_name} is required when the {provider_name} gate is enabled"
        )
    gate_root = Path(gate_root_raw)
    gate_root.mkdir(parents=True, exist_ok=True)
    try:
        wait_limit = float(
            os.environ.get(wait_name, "7200")
        )
    except ValueError as error:
        raise ContractError(
            f"{wait_name} must be numeric"
        ) from error
    if wait_limit <= 0:
        raise ContractError(f"{provider_name} gate wait limit must be positive")






    queue_root = gate_root / "waiters"
    queue_root.mkdir(parents=True, exist_ok=True)
    queue_lock = (gate_root / "queue.lock").open("a+")
    ticket_name = (
        f"{time.time_ns():020d}-{socket.gethostname()}-{os.getpid()}-"
        f"{uuid.uuid4().hex}.ticket"
    )
    ticket_path = queue_root / ticket_name
    ticket_handle = None
    slot_handle = None
    started = time.monotonic()
    try:
        fcntl.flock(queue_lock.fileno(), fcntl.LOCK_EX)
        try:
            ticket_handle = ticket_path.open("x+")
            fcntl.flock(ticket_handle.fileno(), fcntl.LOCK_EX)
            ticket_handle.write(
                json.dumps(
                    {
                        "hostname": socket.gethostname(),
                        "pid": os.getpid(),
                        "queued_unix_seconds": time.time(),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            ticket_handle.flush()
        finally:
            fcntl.flock(queue_lock.fileno(), fcntl.LOCK_UN)

        while True:
            active_limit = limit
            limit_path = gate_root / "active_limit"
            if limit_path.is_file():
                try:
                    active_limit = int(
                        limit_path.read_text(encoding="utf-8").strip()
                    )
                except ValueError as error:
                    raise ContractError(
                        f"{provider_name} gate limit file is not an integer: "
                        f"{limit_path}"
                    ) from error
                if active_limit < 1 or active_limit > limit:
                    raise ContractError(
                        f"{provider_name} gate active limit must be between 1 and the "
                        "configured "
                        f"ceiling of {limit}"
                    )

            fcntl.flock(queue_lock.fileno(), fcntl.LOCK_EX)
            try:
                live_tickets: list[Path] = []
                for queued_path in sorted(queue_root.glob("*.ticket")):
                    if queued_path == ticket_path:
                        live_tickets.append(queued_path)
                        continue
                    queued_handle = queued_path.open("a+")
                    try:
                        fcntl.flock(
                            queued_handle.fileno(),
                            fcntl.LOCK_EX | fcntl.LOCK_NB,
                        )
                    except BlockingIOError:
                        queued_handle.close()
                        live_tickets.append(queued_path)
                        continue
                    queued_path.unlink(missing_ok=True)
                    fcntl.flock(queued_handle.fileno(), fcntl.LOCK_UN)
                    queued_handle.close()

                if live_tickets and live_tickets[0] == ticket_path:
                    for slot in range(active_limit):
                        candidate = (gate_root / f"slot-{slot:03d}.lock").open(
                            "a+"
                        )
                        try:
                            fcntl.flock(
                                candidate.fileno(),
                                fcntl.LOCK_EX | fcntl.LOCK_NB,
                            )
                        except BlockingIOError:
                            candidate.close()
                            continue
                        slot_handle = candidate
                        ticket_path.unlink(missing_ok=True)
                        break
            finally:
                fcntl.flock(queue_lock.fileno(), fcntl.LOCK_UN)

            if slot_handle is not None:
                wait_seconds = time.monotonic() - started
                slot_handle.seek(0)
                slot_handle.truncate()
                slot_handle.write(
                    json.dumps(
                        {
                            "pid": os.getpid(),
                            "slot": slot,
                            "acquired_unix_seconds": time.time(),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                slot_handle.flush()
                yield CodexGateLease(
                    enabled=True,
                    slot=slot,
                    limit=active_limit,
                    wait_seconds=wait_seconds,
                )
                return

            elapsed = time.monotonic() - started
            if elapsed >= wait_limit:
                raise ContractError(
                    f"{provider_name} provider gate wait exceeded "
                    f"{wait_limit:g} seconds"
                )
            time.sleep(0.25)
    finally:
        if slot_handle is not None:
            fcntl.flock(slot_handle.fileno(), fcntl.LOCK_UN)
            slot_handle.close()
        if ticket_handle is not None:
            fcntl.flock(queue_lock.fileno(), fcntl.LOCK_EX)
            try:
                ticket_path.unlink(missing_ok=True)
            finally:
                fcntl.flock(queue_lock.fileno(), fcntl.LOCK_UN)
            fcntl.flock(ticket_handle.fileno(), fcntl.LOCK_UN)
            ticket_handle.close()
        queue_lock.close()


@contextmanager
def codex_concurrency_slot() -> Iterator[CodexGateLease]:
    pass

    with _provider_concurrency_slot(
        provider_name="Codex", environment_prefix="RADONCGYM_CODEX"
    ) as lease:
        yield lease


@contextmanager
def claude_concurrency_slot() -> Iterator[CodexGateLease]:
    pass

    with _provider_concurrency_slot(
        provider_name="Claude", environment_prefix="RADONCGYM_CLAUDE"
    ) as lease:
        yield lease


def codex_transient_backoff_seconds(failed_attempt: int) -> float:
    pass

    return float(min(10 * (2 ** (failed_attempt - 1)), 300))


def is_transient_codex_failure(detail: str) -> bool:
    pass

    normalized = detail.lower()
    markers = (
        "selected model is at capacity",
        "failed to refresh available models",
        "failed to load models cache",
        "timeout waiting for child process",
        "timed out",
        "reading prompt from stdin",
        "failed to clean up stale arg0 temp dirs",
        "resource temporarily unavailable",
        "threadpool failed",
        "connection reset",
        "stream disconnected",
        "error sending request",
        "request failed",
    )
    return any(marker in normalized for marker in markers)
