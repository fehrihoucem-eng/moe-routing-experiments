"""Parse colibri ROUTE_TRACE files into long-form routing records.

Trace grammar (written by c/route_trace.h):

    #prompt <key> <n_prompt_tokens>          -- optional marker, 3 fields
    <call> <row> <layer> <expert>:<gate> ... -- one line per (moe call, batch row)

Lines arrive layer-major within a forward pass: every row of layer 0, then every
row of layer 1, and so on. A forward therefore ends where the layer index stops
increasing, which is how we segment without trusting the global call counter.

Phase assignment uses the marker's ``n_prompt_tokens``: the first that many rows
of a prompt are its prefill, everything after is decode. Deriving it from row
counts instead ("a forward with one row is decode") breaks on a single-token
prompt, and would break again if prefill were ever chunked into several forwards.
Traces with no markers -- anything captured before rt_prompt() existed -- fall
back to "the first forward is the prefill", which is what those runs did.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PREFILL = "prefill"
DECODE = "decode"


@dataclass
class Prompt:
    """One prompt's identity, as recorded in the trace."""

    prompt_id: int
    key: str
    n_prompt_tokens: int
    source: str
    n_decode_tokens: int = 0


@dataclass
class _Forward:
    """One forward pass: rows keyed by position, each holding (layer, ids, gates)."""

    rows: dict[int, list[tuple[int, list[int], list[float]]]] = field(
        default_factory=dict
    )

    def add(self, row: int, layer: int, ids: list[int], gates: list[float]) -> None:
        self.rows.setdefault(row, []).append((layer, ids, gates))

    @property
    def n_rows(self) -> int:
        return len(self.rows)


class TraceFormatError(ValueError):
    """The trace does not match the grammar above."""


def _parse_experts(fields: list[str]) -> tuple[list[int], list[float]]:
    ids: list[int] = []
    gates: list[float] = []
    for tok in fields:
        expert, _, gate = tok.partition(":")
        if not gate:
            raise TraceFormatError(f"expected <expert>:<gate>, got {tok!r}")
        ids.append(int(expert))
        gates.append(float(gate))
    return ids, gates


def parse_trace(
    path: str | Path, first_prompt_id: int = 0
) -> tuple[list[dict], list[Prompt]]:
    """Parse one trace file.

    Returns ``(records, prompts)`` where each record is a dict with the columns
    ``prompt_id, phase, token_id, layer, expert_id, gate`` and ``prompts``
    carries one :class:`Prompt` per prompt seen, in order.

    ``first_prompt_id`` offsets the ids so several files concatenate into one
    store without collisions.
    """
    path = Path(path)
    records: list[dict] = []
    prompts: list[Prompt] = []

    cur: Prompt | None = None
    fwd = _Forward()
    prev_layer = -1
    prev_call: int | None = None
    seen_rows = 0  # rows of this prompt already emitted (prefill accounting)
    decode_idx = 0  # decode forwards seen for this prompt

    def flush() -> None:
        """Emit the buffered forward, assigning phase and token_id."""
        nonlocal fwd, seen_rows, decode_idx, cur
        if not fwd.rows or cur is None:
            fwd = _Forward()
            return

        # A marker-less trace does not state the prompt length; the first forward
        # of the prompt is the prefill, so adopt its width as that length.
        if cur.n_prompt_tokens < 0:
            cur.n_prompt_tokens = fwd.n_rows if seen_rows == 0 else 0

        prefill = seen_rows < cur.n_prompt_tokens
        for row in sorted(fwd.rows):
            if prefill:
                phase, token_id = PREFILL, seen_rows + row
            else:
                phase, token_id = DECODE, decode_idx
            for layer, ids, gates in fwd.rows[row]:
                for expert_id, gate in zip(ids, gates):
                    records.append(
                        {
                            "prompt_id": cur.prompt_id,
                            "phase": phase,
                            "token_id": token_id,
                            "layer": layer,
                            "expert_id": expert_id,
                            "gate": gate,
                        }
                    )
        if prefill:
            seen_rows += fwd.n_rows
        else:
            # Decode emits one token per forward; a wider forward here would mean
            # the prompt length was wrong, so say so rather than mislabel tokens.
            if fwd.n_rows != 1:
                raise TraceFormatError(
                    f"{path}: decode forward for prompt {cur.key!r} has "
                    f"{fwd.n_rows} rows, expected 1"
                )
            decode_idx += 1
            cur.n_decode_tokens = decode_idx
        fwd = _Forward()

    def start_prompt(key: str, ntok: int) -> None:
        nonlocal cur, seen_rows, decode_idx, prev_layer, prev_call
        flush()
        cur = Prompt(
            prompt_id=first_prompt_id + len(prompts),
            key=key,
            n_prompt_tokens=ntok,
            source=path.name,
        )
        prompts.append(cur)
        seen_rows = 0
        decode_idx = 0
        prev_layer = -1
        prev_call = None

    with path.open() as fh:
        for lineno, line in enumerate(fh, 1):
            fields = line.split()
            if not fields:
                continue
            if fields[0] == "#prompt":
                if len(fields) != 3:
                    raise TraceFormatError(
                        f"{path}:{lineno}: malformed marker {line.strip()!r}"
                    )
                start_prompt(fields[1], int(fields[2]))
                continue
            if fields[0].startswith("#"):
                continue  # unknown comment: ignore rather than fail
            if len(fields) < 4:
                continue
            try:
                call, row, layer = int(fields[0]), int(fields[1]), int(fields[2])
                ids, gates = _parse_experts(fields[3:])
            except ValueError as exc:
                raise TraceFormatError(f"{path}:{lineno}: {exc}") from exc

            if cur is None:
                # Data before any marker: a pre-rt_prompt trace, one prompt, and
                # its length is the first forward's width (resolved in flush()).
                start_prompt(key=f"{path.stem}", ntok=-1)

            # Forward boundary. Every row of one layer of one forward shares a
            # call number, so a repeated call is the next *row*, and a new call
            # whose layer did not advance is the next *forward*. Keying on the
            # layer alone ("layer < prev_layer") cannot see the boundary at all
            # when a model has one MoE layer, and mistakes row 1 for a new
            # forward if you weaken it to "<=".
            if call != prev_call:
                if layer <= prev_layer:
                    flush()
                prev_layer = layer
            prev_call = call
            fwd.add(row, layer, ids, gates)

    flush()
    return records, prompts
