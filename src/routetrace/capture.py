"""Drive the qwen36 engine in SERVE mode to capture a multi-prompt trace.

One engine process serves every prompt, so the ~12 s of weight load and VRAM
warmstart is paid once instead of per prompt. Each request emits a ``#prompt``
marker into the trace, which is what makes prompt_id and the prefill/decode
boundary explicit rather than inferred.

Wire protocol (c/qwen36.c, "coli serve mode")::

    engine:  \\x01\\x01READY\\x01\\x01\\n  then  STAT ...
    driver:  SUBMIT <id> <slot> <plen> <max_tok> <temp> <top_p>\\n<payload>\\n
    engine:  ACCEPT <id> <np>  /  DATA <id> <n>\\n<bytes>\\n ...  /  DONE <id> STAT ...
             ERROR <id> <reason ...>
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

READY = b"\x01\x01READY\x01\x01\n"

# np + max_tok must stay under the engine's context (Q36_MAXT, default 8192), or
# serve_one() rejects the request outright -- the same trap that makes a default
# `coli chat` return HTTP 400 on every message.
DEFAULT_MAX_CTX = 8192


@dataclass
class Reply:
    prompt_id: int
    key: str
    prompt: str
    text: str
    n_prompt_tokens: int
    error: str | None = None


class EngineError(RuntimeError):
    pass


def capture(
    prompts: list[str],
    trace_path: str | Path,
    model_dir: str | Path,
    engine: str | Path,
    max_tok: int = 128,
    temp: float = 0.0,
    top_p: float = 1.0,
    cap: int = 256,
    bits: int = 4,
    env_extra: dict[str, str] | None = None,
    timeout: float = 1800.0,
) -> list[Reply]:
    """Run ``prompts`` through one engine process, writing a trace to ``trace_path``."""
    trace_path = Path(trace_path)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    model_dir = Path(model_dir)

    env = dict(os.environ)
    env.update(
        {
            "SERVE": "1",
            "ROUTE_TRACE": str(trace_path),
            "SNAP": str(model_dir),
            "TOK": str(model_dir / "tokenizer.json"),
        }
    )
    if env_extra:
        env.update({k: str(v) for k, v in env_extra.items()})

    proc = subprocess.Popen(
        [str(engine), str(cap), str(bits)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env=env,
        bufsize=0,
    )
    assert proc.stdin and proc.stdout

    def readline() -> bytes:
        line = proc.stdout.readline()
        if not line:
            raise EngineError("engine closed stdout unexpectedly")
        return line

    try:
        while readline() != READY:
            pass
        readline()  # STAT

        replies: list[Reply] = []
        for i, prompt in enumerate(prompts):
            key = f"p{i:05d}"
            payload = prompt.encode()
            header = (
                f"SUBMIT {key} 0 {len(payload)} {max_tok} {temp:.2f} {top_p:.2f}\n"
            ).encode()
            proc.stdin.write(header + payload + b"\n")
            proc.stdin.flush()

            chunks: list[bytes] = []
            np_tokens = 0
            error: str | None = None
            while True:
                line = readline()
                parts = line.split()
                if not parts:
                    continue
                tag = parts[0]
                if tag == b"ACCEPT":
                    np_tokens = int(parts[2])
                elif tag == b"DATA":
                    n = int(parts[2])
                    body = b""
                    while len(body) < n:
                        got = proc.stdout.read(n - len(body))
                        if not got:
                            raise EngineError("engine closed mid-DATA")
                        body += got
                    proc.stdout.read(1)  # trailing newline
                    chunks.append(body)
                elif tag == b"ERROR":
                    error = line.decode(errors="replace").strip()
                    break
                elif tag == b"DONE":
                    break

            replies.append(
                Reply(
                    prompt_id=i,
                    key=key,
                    prompt=prompt,
                    text=b"".join(chunks).decode(errors="replace"),
                    n_prompt_tokens=np_tokens,
                    error=error,
                )
            )

        proc.stdin.close()
        proc.wait(timeout=timeout)
        return replies
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=30)
