#!/usr/bin/env python3
# Copyright (c) Tile-AI Corporation. SPDX-License-Identifier: MIT
"""
Extract and summarize hivm sync_block_wait / sync_block_set from TileLang/HIVM MLIR dumps.

Typical use (after split mix kernel):
  python scripts/cv_sync_report.py softpipeline_test_mlir/generated.mlir \\
      --from-last-marker split-mix-kernel --only-mix

Whole file (may duplicate funcs across IR dumps):
  python scripts/cv_sync_report.py path/to/dump.mlir

Outputs:
  - Per-function ordered list of sync ops (line, kind, core, flag SSA)
  - Histogram for constant flags only (mod 16), comparable to verify-soft-pipeline-sync
  - Side-by-side first-K syncs for *_mix_aic vs *_mix_aiv for quick pairing review


  # 推荐：只看最后一次 Split Mix 之后的 dump，且只看 mix 核函数
    python scripts/cv_sync_report.py softpipeline_test_mlir/generated.mlir ^
    --from-last-marker split-mix-kernel --only-mix
    # 并排对比 AIC / AIV 各自前 32 条 sync（默认 24，可改）
    python scripts/cv_sync_report.py softpipeline_test_mlir/generated.mlir ^
    --from-last-marker split-mix-kernel --only-mix --pairing 32
    # 整份大文件都扫（多段 IR Dump 会重复统计，一般不建议）
    python scripts/cv_sync_report.py softpipeline_test_mlir/generated.mlir --from-last-marker ""
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

FUNC_RE = re.compile(r"^\s*func\.func\s+@(\S+)\s*\(")
# sync_block_wait[<VECTOR>, ...] flag = %35
# sync_block_set[<CUBE>, ...] flag = %37 syn_instr_mode = ...
SYNC_WAIT_RE = re.compile(
    r"hivm\.hir\.sync_block_wait\[\s*<(\w+)>\s*,.*?\]\s*flag\s*=\s*(\S+)"
)
SYNC_SET_RE = re.compile(
    r"hivm\.hir\.sync_block_set\[\s*<(\w+)>\s*,.*?\]\s*flag\s*=\s*(\S+)"
)
CONST_I64_RE = re.compile(
    r"^(\s*)(%\S+)\s*=\s*arith\.constant\s+(-?\d+)\s*:\s*i64"
)
CONST_I32_RE = re.compile(
    r"^(\s*)(%\S+)\s*=\s*arith\.constant\s+(-?\d+)\s*:\s*i32"
)


@dataclass(frozen=True)
class SyncOp:
    line_no: int
    func: str
    kind: str  # "wait" | "set"
    core: str  # VECTOR | CUBE
    flag: str  # SSA name or constant


def slice_from_last_marker(lines: list[str], marker_substr: str | None) -> list[str]:
    if not marker_substr:
        return lines
    hits = [i for i, ln in enumerate(lines) if marker_substr.lower() in ln.lower()]
    if not hits:
        print(f"warning: marker {marker_substr!r} not found; using full file", file=sys.stderr)
        return lines
    start = hits[-1]
    return lines[start:]


def parse_sync_ops(lines: Iterable[str]) -> list[SyncOp]:
    out: list[SyncOp] = []
    current_func = "<toplevel>"
    for i, line in enumerate(lines, start=1):
        m = FUNC_RE.match(line)
        if m:
            current_func = m.group(1)
            continue
        w = SYNC_WAIT_RE.search(line)
        if w:
            out.append(
                SyncOp(i, current_func, "wait", w.group(1), w.group(2).rstrip(","))
            )
            continue
        s = SYNC_SET_RE.search(line)
        if s:
            out.append(
                SyncOp(i, current_func, "set", s.group(1), s.group(2).split()[0])
            )
    return out


def collect_constants(lines: list[str]) -> dict[str, int]:
    """Map %ssa -> int for top-level-ish constants (best-effort across blocks)."""
    m: dict[str, int] = {}
    for line in lines:
        for rx in (CONST_I64_RE, CONST_I32_RE):
            mm = rx.match(line)
            if mm:
                m[mm.group(2)] = int(mm.group(3))
    return m


def flag_mod16_key(flag: str, consts: dict[str, int]) -> str | None:
    if flag.startswith("%"):
        return None
    try:
        v = int(flag)
        return str(v % 16)
    except ValueError:
        pass
    if flag in consts:
        return str(consts[flag] % 16)
    return None


def histogram_constant_flags(
    ops: list[SyncOp], lines: list[str]
) -> tuple[dict[tuple[str, str, str], int], int]:
    """(kind, core, mod16) -> count; return (hist, skipped_dynamic)."""
    consts = collect_constants(lines)
    hist: dict[tuple[str, str, str], int] = defaultdict(int)
    skipped = 0
    for op in ops:
        k = flag_mod16_key(op.flag, consts)
        if k is None:
            skipped += 1
            continue
        hist[(op.kind, op.core, k)] += 1
    return hist, skipped


def print_report(
    ops: list[SyncOp],
    lines: list[str],
    only_mix: bool,
    pairing: int,
) -> None:
    if only_mix:
        ops = [o for o in ops if "_mix_" in o.func]

    by_func: dict[str, list[SyncOp]] = defaultdict(list)
    for o in ops:
        by_func[o.func].append(o)

    print("=== Per-function sync sequence (order = text order in file) ===\n")
    for fn in sorted(by_func.keys()):
        xs = by_func[fn]
        print(f"@{fn}  ({len(xs)} ops)")
        for j, o in enumerate(xs, 1):
            print(f"  {j:4d}  L{o.line_no:5d}  {o.kind:4s}  core=<{o.core}>  flag={o.flag}")
        print()

    hist, dyn = histogram_constant_flags(ops, lines)
    print("=== Histogram (constant / known SSA→constant flags only, bucket = flag mod 16) ===\n")
    if not hist:
        print("  (none — all flags are dynamic SSA)")
    else:
        for (kind, core, m), c in sorted(hist.items(), key=lambda x: (-x[1], x[0])):
            print(f"  {kind:4s}  <{core}>  flag%16={m:>2s}  x{c}")
    print(f"\n  dynamic/unresolved flags: {dyn} op(s)\n")

    # Net set - wait per (core, mod) — same idea as verify-soft-pipeline-sync
    print("=== Net (set - wait) per (core, flag%16), constants only ===\n")
    net: dict[tuple[str, str], int] = defaultdict(int)
    for (kind, core, m), c in hist.items():
        if kind == "set":
            net[(core, m)] += c
        else:
            net[(core, m)] -= c
    for key in sorted(net.keys(), key=lambda k: (k[0], int(k[1]))):
        d = net[key]
        if d != 0:
            print(f"  <{key[0]}> mod16={key[1]:>2s}  net_set_minus_wait={d:+d}")
    if all(v == 0 for v in net.values()):
        print("  (all zero — for this slice and constant flags only)\n")
    else:
        print()

    aic = [o for o in ops if o.func.endswith("_mix_aic")]
    aiv = [o for o in ops if o.func.endswith("_mix_aiv")]
    if aic and aiv and pairing > 0:
        print(f"=== Side-by-side first {pairing} syncs (mix_aic vs mix_aiv) ===\n")
        print(f"{'#':>3}  {'AIC':<52}  {'AIV':<52}")
        for i in range(min(pairing, max(len(aic), len(aiv)))):
            left = ""
            right = ""
            if i < len(aic):
                o = aic[i]
                left = f"L{o.line_no} {o.kind} <{o.core}> {o.flag}"
            if i < len(aiv):
                o = aiv[i]
                right = f"L{o.line_no} {o.kind} <{o.core}> {o.flag}"
            print(f"{i+1:3d}  {left:<52}  {right:<52}")
        print(
            "\nNote: AIC/AIV execute in parallel; text order does not imply runtime order.\n"
            "Use this to spot missing init/set/clear on one side.\n"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mlir", type=argparse.FileType("r", encoding="utf-8"), help="MLIR dump path")
    ap.add_argument(
        "--from-last-marker",
        metavar="SUBSTR",
        default="split-mix-kernel",
        help="Only parse after the last '// -----// IR Dump ...' line containing this "
        "substring (case-insensitive). Use empty string to disable.",
    )
    ap.add_argument(
        "--only-mix",
        action="store_true",
        help="Only include funcs whose name contains _mix_",
    )
    ap.add_argument(
        "--pairing",
        type=int,
        default=24,
        help="Print first N syncs side-by-side for mix_aic vs mix_aiv (0=skip)",
    )
    args = ap.parse_args()
    raw = args.mlir.read().splitlines()
    marker = args.from_last_marker.strip()
    lines = slice_from_last_marker(raw, marker if marker else None)
    ops = parse_sync_ops(lines)
    print_report(ops, lines, args.only_mix, args.pairing)


if __name__ == "__main__":
    main()
