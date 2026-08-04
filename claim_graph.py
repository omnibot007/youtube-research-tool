"""Cross-video claim graph for channel-scale analysis (backlog #2).

Consumes deep-research packages (*_research_package.json) and builds a
claim graph:

- nodes: deduplicated claims -- the same normalized claim across N videos
  becomes ONE node carrying N source citations
- edges: contradictions between nodes (same metric pinned to different
  values), flagged cross_video when the nodes come from different videos
- weights: node frequency = number of distinct videos making the claim

Pure stdlib, no network, safe to import standalone.

Usage (CLI):
    python yt_scrape.py claim-graph PKG_OR_VIDEO_ID [...] [--json] [--output F]
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_WS = re.compile(r"\s+")
_LEAD_ARTICLES = re.compile(r"^(?:the|a|an)\s+", re.IGNORECASE)
_NUM = re.compile(r"\d+(?:\.\d+)?")
_METRIC = re.compile(
    r"(?i)\b(overbought|oversold|support|resistance|stop\s*loss|"
    r"take\s*profit|win\s*rate|rsi|macd|ema|sma|atr)\b"
)


def normalize_claim_text(text: str) -> str:
    """Normalize a matched claim span for cross-video identity matching."""
    t = _WS.sub(" ", (text or "").strip().lower())
    t = _LEAD_ARTICLES.sub("", t)
    return t.rstrip(".!?,;: ")


def claim_node_key(claim: dict) -> str:
    """Node identity: normalized span + sorted claim types."""
    types = ",".join(sorted(claim.get("claim_types") or []))
    return f"{normalize_claim_text(str(claim.get('matched_pattern', '')))}|{types}"


def texts_conflict(
    text_a: str, subject_a: str, text_b: str, subject_b: str
) -> tuple[bool, str, list[str]]:
    """Do two claim texts pin the same metric to different numeric values?

    Mirrors the intra-video numeric-conflict logic in yt_scrape but stays
    self-contained so this module never has to import the big one.
    """
    ta = normalize_claim_text(text_a)
    tb = normalize_claim_text(text_b)
    ma = _METRIC.search((subject_a or "").lower() or ta)
    mb = _METRIC.search((subject_b or "").lower() or tb)
    if not ma or not mb:
        return False, "", []
    metric_a = _WS.sub(" ", ma.group(1).lower())
    metric_b = _WS.sub(" ", mb.group(1).lower())
    if metric_a != metric_b:
        return False, "", []
    na = _NUM.search(ta)
    nb = _NUM.search(tb)
    if not na or not nb:
        return False, "", []
    va, vb = na.group(), nb.group()
    try:
        if float(va) == float(vb):
            return False, "", []
    except ValueError:
        return False, "", []
    reason = f"Same metric '{metric_a}' pinned to different values ({va} vs {vb})"
    return True, reason, [va, vb]


def load_package(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _claims_from_package(pkg: dict) -> tuple[str, str, list[dict]]:
    video = pkg.get("video") or {}
    vid = str(video.get("id", "") or pkg.get("video_id", ""))
    title = str(video.get("title", ""))
    claims = pkg.get("extracted_claims") or []
    return vid, title, [c for c in claims if isinstance(c, dict)]


def build_claim_graph(packages: list[dict]) -> dict:
    """Build the cross-video claim graph from loaded research packages."""
    nodes: dict[str, dict] = {}
    order: list[str] = []
    for pkg in packages:
        vid, title, claims = _claims_from_package(pkg)
        for claim in claims:
            key = claim_node_key(claim)
            if not key.split("|", 1)[0]:
                continue  # empty span, nothing to graph
            node = nodes.get(key)
            if node is None:
                node = {
                    "key": key,
                    "claim": normalize_claim_text(
                        str(claim.get("matched_pattern", ""))),
                    "claim_types": sorted(claim.get("claim_types") or []),
                    "subject": claim.get("subject", ""),
                    "citations": [],
                    "video_ids": [],
                }
                nodes[key] = node
                order.append(key)
            citation = {
                "video_id": vid,
                "video_title": title,
                "sentence": claim.get("sentence", ""),
                "char_offset": claim.get("char_offset", 0),
            }
            if claim.get("start_ts") is not None:
                citation["start_ts"] = claim.get("start_ts")
            if claim.get("deep_link"):
                citation["deep_link"] = claim.get("deep_link")
            node["citations"].append(citation)
            if vid and vid not in node["video_ids"]:
                node["video_ids"].append(vid)

    node_list: list[dict] = []
    for key in order:
        node = nodes[key]
        node["frequency"] = len(node["video_ids"]) or len(node["citations"])
        node_list.append(node)

    edges: list[dict] = []
    for i in range(len(node_list)):
        for j in range(i + 1, len(node_list)):
            a, b = node_list[i], node_list[j]
            conflict, reason, values = texts_conflict(
                a["claim"], a.get("subject", ""),
                b["claim"], b.get("subject", ""),
            )
            if not conflict:
                continue
            cross = set(a["video_ids"]) != set(b["video_ids"])
            edges.append({
                "node_a": a["key"],
                "node_b": b["key"],
                "type": "contradiction",
                "reason": reason,
                "values": values,
                "cross_video": cross,
                "videos_a": list(a["video_ids"]),
                "videos_b": list(b["video_ids"]),
            })

    top = sorted(node_list, key=lambda n: (-n["frequency"], n["claim"]))[:10]
    return {
        "ok": True,
        "mode": "claim_graph",
        "video_count": len({v for n in node_list for v in n["video_ids"]}),
        "node_count": len(node_list),
        "edge_count": len(edges),
        "nodes": node_list,
        "edges": edges,
        "top_claims": [
            {"claim": n["claim"], "frequency": n["frequency"],
             "video_ids": list(n["video_ids"])}
            for n in top
        ],
    }


def build_graph_from_paths(paths: list) -> dict:
    """Load packages from disk and build the graph."""
    return build_claim_graph([load_package(p) for p in paths])
