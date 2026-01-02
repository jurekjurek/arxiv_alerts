#!/usr/bin/env python3
"""
arxiv_alert.py

Keyword-based arXiv alert:
- Fetch recent papers from specific categories
- Filter via include/exclude keywords
- Score + rank
- Deduplicate across runs (seen.json)
- Write a Markdown report (report.md)

Dependencies: requests (pip install requests)
Python: 3.9+
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Iterable, List, Dict, Tuple, Optional

# for the email shenanigans
import os
import smtplib
from email.message import EmailMessage

import requests
import xml.etree.ElementTree as ET

ARXIV_API_URL = "https://export.arxiv.org/api/query"

# ----------------------------
# Configuration (edit these)
# ----------------------------

CATEGORIES = ["cs.AI", "cs.CL", "hep-lat", "quant-ph"]  # Example: machine learning + AI
DAYS_BACK = 20                   # How far back to look
MAX_RESULTS = 300               # Over-fetch, then filter locally

# Keywords are case-insensitive. Use plain phrases; script compiles safe regex.
INCLUDE_ANY = [
    "discocat",
    "krylov",
    "subspace-expansion",
    "wilson",
    "quantum error mitigation",
    "quantum machine learning",
]

# If you want to require multiple terms, use INCLUDE_ALL.
INCLUDE_ALL = [
    #"hybrid", "krylov"
]

EXCLUDE_ANY = [
    #"survey",
]

# Scoring weights
WEIGHT_TITLE_HIT = 3
WEIGHT_ABSTRACT_HIT = 1
WEIGHT_INCLUDE_ALL_BONUS = 2
WEIGHT_EXCLUDE_PENALTY = 9999  # effectively drop

# Files for state/output
STATE_FILE = Path("seen.json")
REPORT_FILE = Path("report.md")


# ----------------------------
# Data model
# ----------------------------

@dataclass
class Paper:
    arxiv_id: str
    title: str
    abstract: str
    authors: List[str]
    categories: List[str]
    published: datetime
    updated: datetime
    link_abs: str
    link_pdf: str


# ----------------------------
# Utility functions
# ----------------------------

def utc_now() -> datetime:
    return datetime.now(timezone.utc)

def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def safe_phrase_regex(phrase: str) -> re.Pattern:
    # Escape all regex metacharacters; treat input as a literal phrase.
    # Match case-insensitively with word-ish boundaries.
    escaped = re.escape(phrase.strip())
    return re.compile(escaped, re.IGNORECASE)

def load_seen_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return set(str(x) for x in data)
        if isinstance(data, dict) and "seen" in data and isinstance(data["seen"], list):
            return set(str(x) for x in data["seen"])
    except Exception:
        pass
    # If corrupted, do not block execution; start fresh.
    return set()

def save_seen_ids(path: Path, seen: set[str]) -> None:
    payload = {"seen": sorted(seen)}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

def build_search_query(categories: List[str]) -> str:
    # arXiv query: cat:cs.LG OR cat:cs.AI
    # Parentheses matter.
    parts = [f"cat:{c}" for c in categories]
    return "(" + " OR ".join(parts) + ")"

def fetch_arxiv_feed(search_query: str, max_results: int) -> str:
    params = {
        "search_query": search_query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    r = requests.get(ARXIV_API_URL, params=params, timeout=30)
    r.raise_for_status()
    return r.text

def parse_atom(feed_xml: str) -> List[Paper]:
    # Atom feed uses namespaces
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }
    root = ET.fromstring(feed_xml)
    papers: List[Paper] = []

    for entry in root.findall("atom:entry", ns):
        arxiv_id_full = entry.findtext("atom:id", default="", namespaces=ns)
        # Example: http://arxiv.org/abs/2401.12345v2
        arxiv_id = arxiv_id_full.rsplit("/", 1)[-1]

        title = normalize_ws(entry.findtext("atom:title", default="", namespaces=ns))
        abstract = normalize_ws(entry.findtext("atom:summary", default="", namespaces=ns))

        published_s = entry.findtext("atom:published", default="", namespaces=ns)
        updated_s = entry.findtext("atom:updated", default="", namespaces=ns)
        published = datetime.fromisoformat(published_s.replace("Z", "+00:00")) if published_s else utc_now()
        updated = datetime.fromisoformat(updated_s.replace("Z", "+00:00")) if updated_s else utc_now()

        authors = []
        for a in entry.findall("atom:author", ns):
            name = a.findtext("atom:name", default="", namespaces=ns)
            if name:
                authors.append(normalize_ws(name))

        categories = []
        for c in entry.findall("atom:category", ns):
            term = c.attrib.get("term")
            if term:
                categories.append(term)

        link_abs = ""
        link_pdf = ""
        for link in entry.findall("atom:link", ns):
            rel = link.attrib.get("rel", "")
            href = link.attrib.get("href", "")
            title_attr = link.attrib.get("title", "")
            if rel == "alternate" and href:
                link_abs = href
            if title_attr.lower() == "pdf" and href:
                link_pdf = href

        papers.append(
            Paper(
                arxiv_id=arxiv_id,
                title=title,
                abstract=abstract,
                authors=authors,
                categories=categories,
                published=published,
                updated=updated,
                link_abs=link_abs or arxiv_id_full,
                link_pdf=link_pdf,
            )
        )

    return papers


# ----------------------------
# Filtering + scoring
# ----------------------------

@dataclass
class MatchResult:
    paper: Paper
    score: int
    include_hits_title: List[str]
    include_hits_abs: List[str]
    include_all_missing: List[str]

def score_and_filter(
    papers: Iterable[Paper],
    include_any: List[str],
    include_all: List[str],
    exclude_any: List[str],
    since: datetime,
    seen_ids: set[str],
) -> List[MatchResult]:
    include_any_rx = [(kw, safe_phrase_regex(kw)) for kw in include_any]
    include_all_rx = [(kw, safe_phrase_regex(kw)) for kw in include_all]
    exclude_rx = [(kw, safe_phrase_regex(kw)) for kw in exclude_any]

    matches: List[MatchResult] = []

    for p in papers:
        # Time window: use published date
        if p.published < since:
            continue

        # Deduplicate
        if p.arxiv_id in seen_ids:
            continue

        hay_title = p.title
        hay_abs = p.abstract

        # Excludes: drop if any match
        excluded = [kw for kw, rx in exclude_rx if rx.search(hay_title) or rx.search(hay_abs)]
        if excluded:
            continue

        # INCLUDE_ALL: all must be present (title or abstract)
        missing_all = []
        for kw, rx in include_all_rx:
            if not (rx.search(hay_title) or rx.search(hay_abs)):
                missing_all.append(kw)
        if include_all and missing_all:
            continue

        # INCLUDE_ANY: at least one must match (unless include_any empty)
        hits_title = [kw for kw, rx in include_any_rx if rx.search(hay_title)]
        hits_abs = [kw for kw, rx in include_any_rx if rx.search(hay_abs)]
        if include_any and not (hits_title or hits_abs):
            continue

        # Score
        score = 0
        score += WEIGHT_TITLE_HIT * len(hits_title)
        score += WEIGHT_ABSTRACT_HIT * len(hits_abs)
        if include_all:
            score += WEIGHT_INCLUDE_ALL_BONUS

        matches.append(
            MatchResult(
                paper=p,
                score=score,
                include_hits_title=hits_title,
                include_hits_abs=hits_abs,
                include_all_missing=missing_all,
            )
        )

    # Rank by score, then newest first
    matches.sort(key=lambda m: (m.score, m.paper.published), reverse=True)
    return matches


# ----------------------------
# Reporting
# ----------------------------

def format_report(matches: List[MatchResult], since: datetime) -> str:
    lines: List[str] = []
    lines.append(f"# arXiv alert report")
    lines.append("")
    lines.append(f"- Generated: {utc_now().strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append(f"- Window: papers published since {since.strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append(f"- Matches: {len(matches)}")
    lines.append("")

    if not matches:
        lines.append("No matches found.")
        lines.append("")
        return "\n".join(lines)

    for i, m in enumerate(matches, start=1):
        p = m.paper
        authors = ", ".join(p.authors[:6]) + (" et al." if len(p.authors) > 6 else "")
        cats = ", ".join(p.categories[:6]) + (" …" if len(p.categories) > 6 else "")

        lines.append(f"## {i}. {p.title}")
        lines.append("")
        lines.append(f"- arXiv: `{p.arxiv_id}`")
        lines.append(f"- Published: {p.published.strftime('%Y-%m-%d %H:%M UTC')}")
        lines.append(f"- Authors: {authors}")
        lines.append(f"- Categories: {cats}")
        lines.append(f"- Links: [abs]({p.link_abs})" + (f" | [pdf]({p.link_pdf})" if p.link_pdf else ""))
        lines.append(f"- Score: {m.score}")
        if m.include_hits_title or m.include_hits_abs:
            lines.append(f"- Keyword hits: title={m.include_hits_title} abstract={m.include_hits_abs}")
        lines.append("")
        # Short abstract excerpt
        excerpt = p.abstract
        if len(excerpt) > 600:
            excerpt = excerpt[:600].rsplit(" ", 1)[0] + "…"
        lines.append(excerpt)
        lines.append("")

    return "\n".join(lines)


# ----------------------------
# Main
# ----------------------------

def main() -> int:
    since = utc_now() - timedelta(days=DAYS_BACK)
    seen_ids = load_seen_ids(STATE_FILE)

    search_query = build_search_query(CATEGORIES)
    try:
        feed_xml = fetch_arxiv_feed(search_query, MAX_RESULTS)
    except requests.RequestException as e:
        print(f"ERROR: failed to fetch arXiv feed: {e}", file=sys.stderr)
        return 2

    try:
        papers = parse_atom(feed_xml)
    except ET.ParseError as e:
        print(f"ERROR: failed to parse arXiv Atom XML: {e}", file=sys.stderr)
        return 3

    matches = score_and_filter(
        papers=papers,
        include_any=INCLUDE_ANY,
        include_all=INCLUDE_ALL,
        exclude_any=EXCLUDE_ANY,
        since=since,
        seen_ids=seen_ids,
    )

    report_md = format_report(matches, since)
    REPORT_FILE.write_text(report_md, encoding="utf-8")

    # Email the report
    subject = f"arXiv Alert — {len(matches)} match(es) — {utc_now().strftime('%Y-%m-%d')}"
    send_email(subject, report_md)

    # Update seen ids with whatever we matched (so we don't resend tomorrow)
    for m in matches:
        seen_ids.add(m.paper.arxiv_id)
    save_seen_ids(STATE_FILE, seen_ids)

    print(f"Wrote {REPORT_FILE} with {len(matches)} matches.")
    return 0


# ----------------------------
# Email sending
# ----------------------------

def send_email(subject: str, body_markdown: str) -> None:
    """
    Sends an email via SMTP using environment variables:
      SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, EMAIL_TO, EMAIL_FROM (optional)

    Notes:
      - Uses STARTTLS.
      - Sends Markdown as plain text (works well in most clients).
    """
    smtp_host = os.environ.get("SMTP_HOST", "").strip()
    smtp_port = int(os.environ.get("SMTP_PORT", "587").strip())
    smtp_user = os.environ.get("SMTP_USER", "").strip()
    smtp_pass = os.environ.get("SMTP_PASS", "").strip()
    email_to = os.environ.get("EMAIL_TO", "").strip()
    email_from = os.environ.get("EMAIL_FROM", "").strip() or smtp_user

    missing = [k for k, v in {
        "SMTP_HOST": smtp_host,
        "SMTP_USER": smtp_user,
        "SMTP_PASS": smtp_pass,
        "EMAIL_TO": email_to,
    }.items() if not v]
    if missing:
        raise RuntimeError(f"Missing required email environment variables: {missing}")

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = email_from
    msg["To"] = email_to
    msg.set_content(body_markdown)

    with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(smtp_user, smtp_pass)
        server.send_message(msg)





if __name__ == "__main__":
    raise SystemExit(main())
