"""Drive the qwen36 engine in SERVE mode to capture a multi-prompt trace.

One engine process serves every prompt, so the ~12 s of weight load and VRAM
warmstart is paid once instead of per prompt. Each request emits a ``#prompt``
marker into the trace, which is what makes prompt_id and the prefill/decode
boundary explicit rather than inferred.

Prompts are rendered through Qwen3.6's chat template before they are submitted.
That is not cosmetic: SERVE takes the payload as raw text and adds nothing, and
the official template's own note is that the model was never trained on a bare
``assistant\\n`` state -- greedy argmax there lands on an EOS special and
generates nothing at all. See :func:`render_chat`.

Wire protocol (c/qwen36.c, "coli serve mode")::

    engine:  \\x01\\x01READY\\x01\\x01\\n  then  STAT ...
    driver:  SUBMIT <id> <slot> <plen> <max_tok> <temp> <top_p>\\n<payload>\\n
    engine:  ACCEPT <id> <np>  /  DATA <id> <n>\\n<bytes>\\n ...  /  DONE <id> STAT ...
             ERROR <id> <reason ...>
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

READY = b"\x01\x01READY\x01\x01\n"

# np + max_tok must stay under the engine's context (Q36_MAXT, default 8192), or
# serve_one() rejects the request outright -- the same trap that makes a default
# `coli chat` return HTTP 400 on every message.
DEFAULT_MAX_CTX = 8192


@dataclass
class PromptSpec:
    text: str
    category: str = ""


@dataclass
class Reply:
    prompt_id: int
    key: str
    category: str
    text: str
    response: str
    n_prompt_tokens: int
    error: str | None = None


class EngineError(RuntimeError):
    pass


def render_chat(
    prompt: str, enable_thinking: bool = False, system: str | None = None
) -> str:
    """Render one user turn with Qwen3.6's chat template.

    Mirrors ``render_chat_qwen`` in c/openai_server.py byte for byte, including
    the trailing think block: enabled leaves it open, disabled pre-closes it so
    the model answers directly. Both branches must emit *something* after
    ``<|im_start|>assistant\\n`` -- that bare state is untrained.
    """
    parts = []
    if system:
        parts.append(f"<|im_start|>system\n{system}<|im_end|>\n")
    parts.append(f"<|im_start|>user\n{prompt}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    parts.append("<think>\n" if enable_thinking else "<think>\n\n</think>\n\n")
    return "".join(parts)


def capture(
    prompts: list[str | PromptSpec | dict],
    trace_path: str | Path,
    model_dir: str | Path,
    engine: str | Path,
    max_tok: int = 200,
    temp: float = 0.0,
    top_p: float = 1.0,
    cap: int = 256,
    bits: int = 4,
    enable_thinking: bool = False,
    system: str | None = None,
    apply_template: bool = True,
    env_extra: dict[str, str] | None = None,
    manifest_path: str | Path | None = None,
    timeout: float = 7200.0,
    progress: bool = False,
) -> list[Reply]:
    """Run ``prompts`` through one engine process, writing a trace to ``trace_path``.

    Writes a manifest JSON alongside the trace (``<trace>.manifest.json`` unless
    ``manifest_path`` says otherwise) carrying each prompt's key, category, text
    and response. The trace itself records only the key, so the manifest is what
    lets :func:`~routetrace.store.build_store` attach categories to X.
    """
    specs: list[PromptSpec] = []
    for p in prompts:
        if isinstance(p, PromptSpec):
            specs.append(p)
        elif isinstance(p, dict):
            specs.append(PromptSpec(text=p["text"], category=p.get("category", "")))
        else:
            specs.append(PromptSpec(text=p))

    trace_path = Path(trace_path)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    model_dir = Path(model_dir)
    if manifest_path is None:
        manifest_path = trace_path.with_suffix(trace_path.suffix + ".manifest.json")
    manifest_path = Path(manifest_path)

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
        for i, spec in enumerate(specs):
            key = f"p{i:05d}"
            rendered = (
                render_chat(spec.text, enable_thinking=enable_thinking, system=system)
                if apply_template
                else spec.text
            )
            payload = rendered.encode()
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

            reply = Reply(
                prompt_id=i,
                key=key,
                category=spec.category,
                text=spec.text,
                response=b"".join(chunks).decode(errors="replace"),
                n_prompt_tokens=np_tokens,
                error=error,
            )
            replies.append(reply)
            if progress:
                mark = "!" if reply.error else "."
                print(
                    f"[{i + 1:>3}/{len(specs)}] {mark} {spec.category:<24} "
                    f"np={reply.n_prompt_tokens:<4} "
                    f"resp={len(reply.response):>5}ch",
                    flush=True,
                )

        proc.stdin.close()
        proc.wait(timeout=timeout)

        manifest = {
            "trace": str(trace_path),
            "max_tok": max_tok,
            "temp": temp,
            "top_p": top_p,
            "enable_thinking": enable_thinking,
            "apply_template": apply_template,
            "system": system,
            "prompts": [asdict(r) for r in replies],
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        return replies
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=30)
