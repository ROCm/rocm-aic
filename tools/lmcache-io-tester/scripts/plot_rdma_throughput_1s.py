#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
"""Plot RDMA rx/tx throughput (SI GB/s) in 1-second bins from rdma-statistic logs.

Samples are taken every few seconds in typical bench scripts; this tool
linearly interpolates cumulative rx_bytes / tx_bytes between sample times,
then differences the interpolated curve at 1 Hz to approximate bytes moved
each second (SI GB/s = 1e9 bytes per second).

Example:
  ./scripts/plot_rdma_throughput_1s.py \\
    --input /tmp/lmcache-nfs-rdma-fetched/g04u07/rdma-statistic.sample.log \\
    --iface rocep159s0 \\
    --output /tmp/rdma-gbs-1s.svg
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Sequence, Tuple


@dataclass(frozen=True)
class Sample:
    t: float
    rx_bytes: int
    tx_bytes: int


def _parse_samples(text: str, iface: str) -> List[Sample]:
    parts = text.split("=====")
    out: List[Sample] = []
    needle = f"link {iface}/1"
    for i in range(1, len(parts) - 1, 2):
        ts_raw = parts[i].strip()
        body = parts[i + 1]
        if not re.match(r"\d{4}-\d{2}-\d{2}", ts_raw):
            continue
        pos = body.find(needle)
        if pos < 0:
            continue
        seg = body[pos : pos + 4000]
        m = re.search(r"rx_bytes (\d+).*?tx_bytes (\d+)", seg, re.S)
        if not m:
            continue
        try:
            ts = datetime.fromisoformat(ts_raw)
        except ValueError:
            continue
        out.append(
            Sample(
                float(ts.timestamp()),
                int(m.group(1)),
                int(m.group(2)),
            )
        )
    out.sort(key=lambda s: s.t)
    return out


def _lerp_bytes(samples: Sequence[Sample], t: float, field: str) -> float:
    if not samples:
        return 0.0
    if t <= samples[0].t:
        v = getattr(samples[0], field)
        return float(v)
    if t >= samples[-1].t:
        v = getattr(samples[-1], field)
        return float(v)
    for i in range(len(samples) - 1):
        t0, t1 = samples[i].t, samples[i + 1].t
        if t0 <= t <= t1:
            v0 = float(getattr(samples[i], field))
            v1 = float(getattr(samples[i + 1], field))
            if t1 <= t0:
                return v0
            return v0 + (v1 - v0) * (t - t0) / (t1 - t0)
    return float(getattr(samples[-1], field))


def _one_hz_gb_per_s(samples: Sequence[Sample]) -> Tuple[float, List[float], List[float]]:
    """Return (t0_unix, rx_GBps_each_sec, tx_GBps_each_sec)."""
    if len(samples) < 2:
        return samples[0].t if samples else 0.0, [], []
    t0 = samples[0].t
    t1 = samples[-1].t
    n = max(0, int(math.floor(t1 - t0)))
    rx_rates: List[float] = []
    tx_rates: List[float] = []
    for k in range(n):
        ta = t0 + k
        tb = t0 + k + 1
        rx_a = _lerp_bytes(samples, ta, "rx_bytes")
        rx_b = _lerp_bytes(samples, tb, "rx_bytes")
        tx_a = _lerp_bytes(samples, ta, "tx_bytes")
        tx_b = _lerp_bytes(samples, tb, "tx_bytes")
        rx_rates.append((rx_b - rx_a) / 1.0e9)
        tx_rates.append((tx_b - tx_a) / 1.0e9)
    return t0, rx_rates, tx_rates


def _svg_polyline(
    xs: Sequence[float],
    ys: Sequence[float],
    y_max: float,
    width: int,
    height: int,
    margin: int,
    stroke: str,
) -> str:
    if not xs or y_max <= 0:
        return ""
    inner_w = width - 2 * margin
    inner_h = height - 2 * margin
    pts = []
    for x, y in zip(xs, ys):
        px = margin + (x / max(xs[-1], 1e-9)) * inner_w
        py = margin + inner_h - (y / y_max) * inner_h
        pts.append(f"{px:.1f},{py:.1f}")
    return (
        f'<polyline fill="none" stroke="{stroke}" stroke-width="2" '
        f'points="{" ".join(pts)}" />'
    )


def write_svg(
    path: Path,
    rx_gbps: Sequence[float],
    tx_gbps: Sequence[float],
    title: str,
) -> None:
    n = len(rx_gbps)
    xs = [float(i) for i in range(n)]
    y_max = max(
        max(rx_gbps) if rx_gbps else 0.0,
        max(tx_gbps) if tx_gbps else 0.0,
        1e-6,
    ) * 1.08
    w, h, m = 1100, 420, 55
    rx_line = _svg_polyline(xs, rx_gbps, y_max, w, h, m, "#2563eb")
    tx_line = _svg_polyline(xs, tx_gbps, y_max, w, h, m, "#d97706")
    # Y-axis ticks (5)
    ticks = []
    for i in range(6):
        gv = y_max * i / 5
        yp = m + (h - 2 * m) * (1.0 - i / 5)
        ticks.append(
            f'<text x="{m - 8}" y="{yp + 4}" text-anchor="end" '
            f'font-size="11" fill="#444">{gv:.3f}</text>'
        )
        ticks.append(
            f'<line x1="{m}" y1="{yp}" x2="{w - m}" y2="{yp}" '
            'stroke="#ddd" stroke-width="1"/>'
        )
    svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
  <rect width="100%" height="100%" fill="#fafafa"/>
  <text x="{m}" y="32" font-size="18" font-weight="600" fill="#111">{title}</text>
  <text x="{m}" y="52" font-size="12" fill="#555">RX / TX throughput (SI GB/s); 1 s bins via linear interpolation of cumulative counters</text>
  {"".join(ticks)}
  <line x1="{m}" y1="{m}" x2="{m}" y2="{h - m}" stroke="#333" stroke-width="1.5"/>
  <line x1="{m}" y1="{h - m}" x2="{w - m}" y2="{h - m}" stroke="#333" stroke-width="1.5"/>
  {rx_line}
  {tx_line}
  <text x="{m}" y="{h - m + 38}" font-size="12" fill="#111">Time (s from first sample)</text>
  <text transform="translate(18 {h // 2}) rotate(-90)" font-size="12" fill="#111">GB/s</text>
  <rect x="{w - 220}" y="68" width="200" height="52" fill="#fff" stroke="#ccc"/>
  <line x1="{w - 200}" y1="88" x2="{w - 170}" y2="88" stroke="#2563eb" stroke-width="3"/>
  <text x="{w - 160}" y="92" font-size="12" fill="#111">RX GB/s</text>
  <line x1="{w - 200}" y1="108" x2="{w - 170}" y2="108" stroke="#d97706" stroke-width="3"/>
  <text x="{w - 160}" y="112" font-size="12" fill="#111">TX GB/s</text>
</svg>
'''
    path.write_text(svg, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Path to rdma-statistic.sample.log (===== timestamp ===== blocks)",
    )
    ap.add_argument(
        "--iface",
        default="rocep159s0",
        help="rdma link name as in 'link <iface>/1' (default: rocep159s0)",
    )
    ap.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output .svg path",
    )
    args = ap.parse_args()
    text = args.input.read_text(encoding="utf-8", errors="replace")
    samples = _parse_samples(text, args.iface)
    if len(samples) < 2:
        raise SystemExit(
            f"Need at least 2 samples with {args.iface}; found {len(samples)}"
        )
    t0, rx_gbps, tx_gbps = _one_hz_gb_per_s(samples)
    title = (
        f"RDMA {args.iface} throughput (1 s, SI GB/s) — "
        f"{len(rx_gbps)} s span, {len(samples)} raw samples"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_svg(args.output, rx_gbps, tx_gbps, title)
    peak_rx = max(rx_gbps) if rx_gbps else 0.0
    peak_tx = max(tx_gbps) if tx_gbps else 0.0
    print(
        f"Wrote {args.output} ({len(rx_gbps)} points). "
        f"Peak RX {peak_rx:.3f} GB/s, peak TX {peak_tx:.3f} GB/s."
    )


if __name__ == "__main__":
    main()
