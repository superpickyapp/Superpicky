# -*- coding: utf-8 -*-
"""
HTTP source probe helpers for initialization.

Notes:
- We intentionally do not use ICMP ping as the primary selection mechanism.
- Some networks block ping while HTTPS still works normally.
- Selection is based on real HTTP responsiveness and cached for the current run.
"""

from __future__ import annotations

import time
import urllib.request
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional


DEFAULT_TIMEOUT_SECONDS = 4.0


@dataclass
class ProbeResult:
    name: str
    url: str
    ok: bool
    total_ms: float
    first_byte_ms: float
    error: Optional[str] = None


_PROBE_CACHE: Dict[str, List[ProbeResult]] = {}


def _normalize_probe_url(url: str) -> str:
    if url.endswith("/simple"):
        return url.rstrip("/") + "/pip/"
    return url


def probe_url(name: str, url: str, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> ProbeResult:
    start = time.perf_counter()
    request = urllib.request.Request(
        _normalize_probe_url(url),
        headers={"User-Agent": "SuperPicky-InitProbe/1.0"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            first_byte_start = time.perf_counter()
            response.read(256)
            first_byte_ms = (time.perf_counter() - first_byte_start) * 1000.0
        total_ms = (time.perf_counter() - start) * 1000.0
        return ProbeResult(
            name=name,
            url=url,
            ok=True,
            total_ms=total_ms,
            first_byte_ms=first_byte_ms,
        )
    except Exception as exc:
        total_ms = (time.perf_counter() - start) * 1000.0
        return ProbeResult(
            name=name,
            url=url,
            ok=False,
            total_ms=total_ms,
            first_byte_ms=0.0,
            error=f"{type(exc).__name__}: {exc}",
        )


def probe_sources(group_name: str, sources: Iterable[dict], timeout: float = DEFAULT_TIMEOUT_SECONDS) -> List[ProbeResult]:
    if group_name in _PROBE_CACHE:
        return list(_PROBE_CACHE[group_name])

    results: List[ProbeResult] = []
    for source in sources:
        results.append(probe_url(source["name"], source["url"], timeout=timeout))
    _PROBE_CACHE[group_name] = list(results)
    return results


def pick_best_source(results: Iterable[ProbeResult]) -> Optional[ProbeResult]:
    successful = [item for item in results if item.ok]
    if not successful:
        return None
    return min(successful, key=lambda item: (item.total_ms, item.first_byte_ms))


def clear_probe_cache() -> None:
    _PROBE_CACHE.clear()
