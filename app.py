"""
Multi-stage cognitive pipeline (Streamlit + Google Gemini via OpenAI-compatible API)

Stage 1  Deep Thinking (First Principles)   4 sequential phases
Stage 2  Radical Ideation (Baseline Burner) 4 sequential steps
Stage 3  The Triad Debate                   Proponent | Inquisitor | Contrarian (parallel) -> Arbiter
Stage 4  Review & Synthesis                 2 Reviewers (parallel) -> Final Writer

Setup
-----
    pip install streamlit openai requests ddgs
    # (the older package name `duckduckgo_search` also works)

Keys: put API_KEY_1, API_KEY_2, ... (up to API_KEY_10) in .streamlit/secrets.toml or in
environment variables. Every sub-agent LLM call rotates to the next key.

Run:  streamlit run app.py
"""

from __future__ import annotations

import concurrent.futures as cf
import itertools
import json
import os
import random
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from urllib.parse import urlparse

import requests
import streamlit as st
from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)

# ════════════════════════════════════════════════════════════════════════════
# 1. CONFIG
# ════════════════════════════════════════════════════════════════════════════

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
MODEL_CHOICES = [
    "gemini-3.8-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash",  # closed to new users (404)
]
KEY_NAMES = tuple(f"API_KEY_{i}" for i in range(1, 11))  # API_KEY_1 ... API_KEY_10

RPM_PER_KEY = 10            # free-tier limit per key (check AI Studio; raise if you enable billing)
MAX_ATTEMPTS = 8            # per LLM call (each attempt rotates to a fresh key)
MAX_SEARCHES_PER_AGENT = 2  # search queries one agent may trigger
SEARCH_RESULTS = 3          # pages pulled per query
PAGE_CHARS = 2500           # chars kept per page
CONTEXT_CLIP = 14000        # chars of any prior output passed forward


class PipelineError(RuntimeError):
    """Raised when an LLM call cannot be completed after all retries."""


@dataclass
class Config:
    model: str
    instructions: str = ""
    personalization: str = ""
    search_enabled: bool = True
    rpm: int = RPM_PER_KEY
    token_budget: int = 500_000
    tavily_key: str = ""
    jina_key: str = ""
    research_enabled: bool = True
    use_grounding: bool = False
    evidence: str = ""  # filled in by Stage 0, injected into every later agent


# ════════════════════════════════════════════════════════════════════════════
# 2. API KEY ROTATION + RATE-LIMIT-AWARE LLM CALLS
# ════════════════════════════════════════════════════════════════════════════

def load_key_items() -> list[tuple[str, str]]:
    """Read API_KEY_1..10 from Streamlit secrets or environment variables."""
    items: list[tuple[str, str]] = []
    for name in KEY_NAMES:
        value = None
        try:
            value = st.secrets.get(name)
        except Exception:  # no secrets.toml present
            value = None
        value = value or os.environ.get(name)
        if value:
            items.append((name, str(value).strip().strip('"').strip("'").strip()))
    return items


def load_keys() -> list[str]:
    return [v for _, v in load_key_items()]


class KeyRotator:
    """
    Thread-safe round-robin over the Gemini keys (itertools.cycle).

    acquire() hands out the next key AND reserves a time slot on it so that no
    single key exceeds RPM_PER_KEY, even with parallel agents.
    """

    def __init__(
        self, keys: list[str], rpm: int = RPM_PER_KEY, labels: dict[str, str] | None = None
    ):
        self._labels = labels or {}
        self.bad: set[str] = set()  # NAMES (never values) of keys Google rejected
        self._keys = list(keys)
        self._cycle = itertools.cycle(keys)
        self._dead_until = {k: 0.0 for k in keys}
        self._lock = threading.Lock()
        self._gap = 60.0 / rpm + 0.35  # small safety margin
        self._next_free = {k: 0.0 for k in keys}

    def acquire(self) -> str:
        with self._lock:
            now = time.monotonic()
            for _ in range(len(self._keys)):
                key = next(self._cycle)
                if self._dead_until[key] <= now:
                    break
            else:
                raise PipelineError(
                    "Every API key is either out of daily quota or was rejected as invalid"
                    + (f" ({', '.join(sorted(self.bad))})" if self.bad else "") + ". "
                    "Quotas reset at midnight Pacific time. Try a lighter model such as "
                    "gemini-3.5-flash-lite, add keys from more Google projects, or wait."
                )
            start = max(now, self._next_free[key])
            self._next_free[key] = start + self._gap
            wait = start - now
        if wait > 0:
            time.sleep(wait)
        return key

    def report_bad(self, key: str) -> None:
        """Google said this key is invalid: remember its name and stop using it."""
        with self._lock:
            self.bad.add(self._labels.get(key, f"key #{self._keys.index(key) + 1}"))
            self._dead_until[key] = time.monotonic() + 86400

    def mark_dead(self, key: str, seconds: float = 3600) -> None:
        """Skip a key whose daily quota is used up."""
        with self._lock:
            self._dead_until[key] = time.monotonic() + seconds

    def penalize(self, key: str, seconds: float) -> None:
        """Rest a key that just returned 429."""
        with self._lock:
            self._next_free[key] = max(self._next_free[key], time.monotonic() + seconds)


ROTATOR_VERSION = 3  # bump this whenever KeyRotator changes, so a stale cached copy is dropped


@st.cache_resource
def get_rotator(items: tuple, rpm: int, version: int) -> KeyRotator:
    # cached so throttle state survives Streamlit reruns; items = ((name, key), ...)
    return KeyRotator([v for _, v in items], rpm, {v: n for n, v in items})


RETRYABLE = {408, 409, 429, 500, 502, 503, 504}


class LiveBuffer:
    """Text streamed in by a worker thread and read by the UI (main) thread."""

    def __init__(self):
        self._lock = threading.Lock()
        self._text = ""
        self._mark = 0
        self.version = 0

    def append(self, s: str) -> None:
        with self._lock:
            self._text += s
            self.version += 1

    def set(self, s: str) -> None:
        with self._lock:
            self._text = s
            self._mark = len(s)
            self.version += 1

    def mark(self) -> None:
        with self._lock:
            self._mark = len(self._text)

    def rollback(self) -> None:
        """Drop a half-streamed answer before a retry."""
        with self._lock:
            self._text = self._text[: self._mark]
            self.version += 1

    def snapshot(self) -> str:
        with self._lock:
            return self._text


class Usage:
    """Thread-safe token counters."""

    def __init__(self):
        self._lock = threading.Lock()
        self.prompt = 0
        self.completion = 0
        self.calls = 0
        self.estimated = False  # True if any count was a chars/4 guess

    def add(self, prompt: int, completion: int, estimated: bool = False) -> None:
        with self._lock:
            self.prompt += int(prompt)
            self.completion += int(completion)
            self.calls += 1
            self.estimated = self.estimated or estimated

    def merge(self, other: "Usage") -> None:
        with self._lock:
            self.prompt += other.prompt
            self.completion += other.completion
            self.calls += other.calls
            self.estimated = self.estimated or other.estimated

    @property
    def total(self) -> int:
        return self.prompt + self.completion


def _err_detail(e: Exception) -> str:
    try:
        return e.response.text[:800]
    except Exception:
        return str(e)[:800]


def _retry_delay(detail: str) -> float | None:
    """Google often says 'retry in 23s' or "retryDelay": "23s"."""
    m = re.search(r'retryDelay"?\s*:\s*"?(\d+(?:\.\d+)?)s', detail) or re.search(
        r"retry in (\d+(?:\.\d+)?)s", detail, re.I
    )
    return float(m.group(1)) if m else None


_FLAGS = {"stream_usage": True}  # switched off automatically if Google rejects it


def call_llm(
    rot: KeyRotator,
    model: str,
    system: str,
    user: str,
    temperature: float = 0.6,
    max_tokens: int = 8192,
    buf: LiveBuffer | None = None,
    usage: Usage | None = None,
) -> str:
    """One streamed chat completion. Rotates to a fresh key on every attempt."""
    last_err: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        key = rot.acquire()
        if buf:
            buf.mark()
        try:
            client = OpenAI(
                api_key=key, base_url=GEMINI_BASE_URL, timeout=180, max_retries=0
            )
            kwargs = dict(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            if _FLAGS["stream_usage"]:
                kwargs["stream_options"] = {"include_usage": True}
            stream = client.chat.completions.create(**kwargs)

            parts: list[str] = []
            p_tok = c_tok = t_tok = None
            for chunk in stream:
                u = getattr(chunk, "usage", None)
                if u:
                    p_tok = getattr(u, "prompt_tokens", None)
                    c_tok = getattr(u, "completion_tokens", None)
                    t_tok = getattr(u, "total_tokens", None)
                if not chunk.choices:
                    continue
                piece = chunk.choices[0].delta.content
                if piece:
                    parts.append(piece)
                    if buf:
                        buf.append(piece)

            text = "".join(parts).strip()
            if text:
                if usage is not None:
                    if p_tok is None and c_tok is None and t_tok is None:
                        usage.add(len(system + user) // 4, len(text) // 4, estimated=True)
                    else:
                        p = p_tok or 0
                        c = (t_tok - p) if (t_tok is not None and t_tok >= p) else (c_tok or 0)
                        usage.add(p, c)
                return text
            last_err = RuntimeError("Model returned an empty response")
            if buf:
                buf.rollback()
            time.sleep(1.5)
        except RateLimitError as e:  # 429
            detail = _err_detail(e)
            last_err = RuntimeError(f"429 rate limit: {detail}")
            if buf:
                buf.rollback()
            if re.search(r"per ?day|daily", detail, re.I):
                rot.mark_dead(key)  # this key is finished for today; use the others
                continue
            delay = _retry_delay(detail) or min(5 * 2 ** attempt, 60)
            rot.penalize(key, delay)
            time.sleep(min(delay, 30) + random.random())
        except APIStatusError as e:
            last_err = e
            if buf:
                buf.rollback()
            detail0 = _err_detail(e)
            if e.status_code in (401, 403) or (
                e.status_code == 400 and re.search(r"api key not valid|invalid api key", detail0, re.I)
            ):
                rot.report_bad(key)  # skip this key, keep going with the others
                last_err = RuntimeError(f"{e.status_code}: {detail0}")
                continue
            if e.status_code == 400 and _FLAGS["stream_usage"]:
                _FLAGS["stream_usage"] = False  # retry once without usage reporting
                continue
            if e.status_code not in RETRYABLE:
                detail = _err_detail(e)
                raise PipelineError(
                    f"API error {e.status_code} (model: {model}): {detail}"
                ) from e
            last_err = RuntimeError(f"{e.status_code}: {_err_detail(e)}")
            time.sleep(min(2 ** attempt, 16) + random.random())
        except (APIConnectionError, APITimeoutError, APIError) as e:
            last_err = e
            if buf:
                buf.rollback()
            time.sleep(min(2 ** attempt, 16) + random.random())
    raise PipelineError(f"LLM call failed after {MAX_ATTEMPTS} attempts: {last_err}")


# ════════════════════════════════════════════════════════════════════════════
# 3. LIVE RESEARCH ENGINE
#    search (Tavily or DuckDuckGo) → Reddit thread reader (with upvote scores)
#    → Jina Reader for other pages → optional Google Search grounding
# ════════════════════════════════════════════════════════════════════════════

MAX_REDDIT_THREADS = 6
MAX_OTHER_PAGES = 5
EVIDENCE_CHARS = 48000
REDDIT_RE = re.compile(
    r"^https?://(?:www\.|old\.|np\.|new\.)?reddit\.com/r/[^/]+/comments/[^/?#]+", re.I
)
NO_EVIDENCE = (
    "(No live sources could be retrieved for this problem. Rely on reasoning, and say so "
    "wherever real-world evidence would matter.)"
)


def _secret(name: str) -> str:
    """Read an optional secret (Streamlit secrets or environment). Call on the main thread."""
    value = None
    try:
        value = st.secrets.get(name)
    except Exception:
        value = None
    value = value or os.environ.get(name)
    return str(value).strip().strip('"').strip("'").strip() if value else ""


def _norm_url(url: str) -> str:
    try:
        u = urlparse((url or "").strip())
    except Exception:
        return ""
    if not u.scheme.startswith("http") or not u.netloc:
        return ""
    host = u.netloc.lower()
    if host.endswith("reddit.com"):
        return f"https://www.reddit.com{u.path.rstrip('/')}"
    return u._replace(fragment="").geturl()


def _ddg_results(query: str, n: int) -> list[dict]:
    try:
        from ddgs import DDGS  # newer package name
    except ImportError:
        from duckduckgo_search import DDGS  # legacy name
    with DDGS() as d:
        return list(d.text(query, max_results=n))


def _tavily_search(key: str, query: str, n: int, domains: list[str] | None) -> list[dict]:
    payload = {"query": query, "max_results": n, "search_depth": "basic", "topic": "general"}
    if domains:
        payload["include_domains"] = domains
    r = requests.post(
        "https://api.tavily.com/search", json=payload,
        headers={"Authorization": f"Bearer {key}"}, timeout=25,
    )
    r.raise_for_status()
    return [
        {"title": x.get("title", ""), "url": x.get("url", ""), "snippet": x.get("content", "")}
        for x in r.json().get("results", [])
    ]


def search_backend(
    cfg: Config, query: str, n: int = 6, domains: list[str] | None = None
) -> tuple[list[dict], str]:
    """Tavily first (built for AI agents, can be pinned to reddit.com); DuckDuckGo as fallback."""
    note = ""
    if cfg.tavily_key:
        try:
            hits = _tavily_search(cfg.tavily_key, query, n, domains)
            if hits:
                return hits, ""
        except Exception as e:
            note = f"Tavily failed ({type(e).__name__}); fell back to DuckDuckGo"
    try:
        q = query
        if domains:
            q += " " + " OR ".join(f"site:{d}" for d in domains)
        raw = _ddg_results(q, n)
        return [
            {"title": x.get("title", ""), "url": x.get("href", ""), "snippet": x.get("body", "")}
            for x in raw
        ], note
    except Exception as e:
        return [], f"search failed: {type(e).__name__}"


def _jina_fetch(url: str, max_chars: int = PAGE_CHARS, key: str = "") -> str:
    headers = {"Accept": "text/plain", "X-Return-Format": "markdown"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    r = requests.get(f"https://r.jina.ai/{url}", headers=headers, timeout=25)
    r.raise_for_status()
    return re.sub(r"\n{3,}", "\n\n", r.text).strip()[:max_chars]


def fetch_reddit_thread(url: str, max_chars: int = 7000, jina_key: str = "") -> str:
    """
    Read a Reddit thread WITH upvote scores so real consensus can be weighed.
    Reddit's public JSON is tried first (often blocked on cloud servers); Jina Reader is the fallback.
    """
    clean = _norm_url(url)
    try:
        r = requests.get(
            clean + ".json",
            params={"limit": 80, "sort": "top", "raw_json": 1},
            headers={"User-Agent": "Mozilla/5.0 (compatible; research-assistant/1.0)"},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        post = data[0]["data"]["children"][0]["data"]
        out = [
            f"r/{post.get('subreddit')} · {post.get('title')} "
            f"(score {post.get('score')}, {post.get('num_comments')} comments)"
        ]
        body = (post.get("selftext") or "").strip()
        if body:
            out.append("POST: " + body[:1800])
        comments = [c["data"] for c in data[1]["data"]["children"] if c.get("kind") == "t1"]
        comments.sort(key=lambda c: c.get("score", 0), reverse=True)
        out.append("TOP COMMENTS (highest upvotes first):")
        for c in comments[:14]:
            txt = re.sub(r"\s+", " ", c.get("body", "")).strip()
            if txt and txt not in ("[deleted]", "[removed]"):
                out.append(f"- (+{c.get('score', 0)}) {txt[:700]}")
        text = "\n".join(out)
        if len(text) > 300:
            return text[:max_chars]
    except Exception:
        pass
    return _jina_fetch(clean, max_chars, jina_key)


def grounded_search(rot: KeyRotator, model: str, query: str) -> tuple[str, list, str]:
    """
    Gemini + Google Search grounding through the native API (this is what AI Studio uses).
    Free API keys usually can't use it; it falls back silently. Returns (text, links, note).
    """
    if not _FLAGS.get("grounding", True):
        return "", [], ""
    key = rot.acquire()
    body = {
        "contents": [{"role": "user", "parts": [{"text": (
            "Research this using web search. Prefer first-hand experiences from real people "
            "(Reddit, forums, reviews) over marketing pages. Report concrete findings, numbers "
            "and disagreements, and say where each came from.\n\nQuery: " + query
        )}]}],
        "tools": [{"google_search": {}}],
        "generationConfig": {"temperature": 0.2},
    }
    try:
        r = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            headers={"x-goog-api-key": key, "Content-Type": "application/json"},
            json=body, timeout=90,
        )
    except requests.RequestException as e:
        return "", [], f"Google grounding unreachable ({type(e).__name__})"
    if r.status_code != 200:
        if r.status_code == 401 or "API key not valid" in r.text:
            rot.report_bad(key)
        else:
            _FLAGS["grounding"] = False  # free keys / unsupported model: stop trying this run
        return "", [], (
            f"Google Search grounding isn't available with these keys ({r.status_code}); "
            "used other search instead"
        )
    try:
        cand = r.json()["candidates"][0]
        text = "".join(p.get("text", "") for p in cand["content"]["parts"]).strip()
        chunks = cand.get("groundingMetadata", {}).get("groundingChunks", [])
        links = [(c["web"].get("title", ""), c["web"].get("uri", "")) for c in chunks if "web" in c]
    except Exception:
        return "", [], "Google grounding returned an unexpected reply"
    return text, links, ""


def search_web(query: str, cfg: Config | None = None, n_results: int = SEARCH_RESULTS) -> str:
    """
    On-demand search used by agents. Never raises.
    If the query contains the word 'reddit', the search is pinned to reddit.com and
    threads are read with their upvote scores.
    """
    cfg = cfg or Config(model="")
    wants_reddit = "reddit" in query.lower()
    q = query
    if wants_reddit:
        q = re.sub(r"(?i)site:reddit\.com|\breddit\b", " ", query).strip() or query
    hits, note = search_backend(
        cfg, q, n_results + (3 if wants_reddit else 0), ["reddit.com"] if wants_reddit else None
    )
    if not hits:
        return f"(no results{'; ' + note if note else ''})"

    def read(h: dict) -> str:
        url = _norm_url(h["url"])
        m = REDDIT_RE.match(url)
        try:
            body = (
                fetch_reddit_thread(m.group(0), 3500, cfg.jina_key) if m
                else _jina_fetch(url, PAGE_CHARS, cfg.jina_key)
            )
        except Exception:
            body = ""
        if len(body) < 200:
            body = h.get("snippet", "")
        return f"[{h['title']}]({url})\n{body}"

    with cf.ThreadPoolExecutor(max_workers=min(4, len(hits))) as pool:
        pages = list(pool.map(read, hits[: n_results + (3 if wants_reddit else 0)]))
    return "\n\n---\n\n".join(pages)


def gather_sources(cfg: Config, rot: KeyRotator, queries: list[tuple[str, str]], buf) -> tuple[str, list[dict]]:
    """Search every planned query, then READ the best threads/pages. Streams progress into buf."""
    _FLAGS["grounding"] = True

    def log(s: str) -> None:
        buf.append(s + "\n")

    cands: list[dict] = []
    extra: list[dict] = []
    seen: set[str] = set()
    notes: list[str] = []

    for kind, q in queries:
        log(f"🔎 {kind}: {q}")
        hits, note = search_backend(
            cfg, q, 8 if kind == "REDDIT" else 5, ["reddit.com"] if kind == "REDDIT" else None
        )
        if note and note not in notes:
            notes.append(note)
        for h in hits:
            url = _norm_url(h["url"])
            m = REDDIT_RE.match(url)
            if m:
                url, k = m.group(0), "reddit"
            elif kind == "REDDIT":
                continue  # a Reddit query should only keep real Reddit threads
            else:
                k = "community" if kind == "COMMUNITY" else "web"
            if not url or url in seen:
                continue
            seen.add(url)
            cands.append({**h, "url": url, "kind": k})
        if cfg.use_grounding and kind != "REDDIT":
            text, links, gnote = grounded_search(rot, cfg.model, q)
            if gnote and gnote not in notes:
                notes.append(gnote)
            if text:
                used = "; ".join(t for t, _ in links[:5] if t)
                extra.append({
                    "title": f"Google Search grounded answer — {q}",
                    "url": links[0][1] if links else "",
                    "kind": "google",
                    "text": text[:4500] + (f"\n(Pages it used: {used})" if used else ""),
                })
        time.sleep(0.3)

    reddit = [c for c in cands if c["kind"] == "reddit"][:MAX_REDDIT_THREADS]
    others = [c for c in cands if c["kind"] != "reddit"]
    others.sort(key=lambda c: c["kind"] != "community")  # forums before generic web
    chosen = reddit + others[:MAX_OTHER_PAGES]
    if not reddit:
        log("⚠️ No Reddit threads found for these queries.")
    log(f"📖 Reading {len(chosen)} pages…")

    def read(c: dict) -> dict:
        try:
            text = (
                fetch_reddit_thread(c["url"], 7000, cfg.jina_key) if c["kind"] == "reddit"
                else _jina_fetch(c["url"], 4500, cfg.jina_key)
            )
        except Exception:
            text = ""
        if len(text) < 200:
            text = c.get("snippet", "")
        return {**c, "text": text}

    read_ok: dict[int, dict] = {}
    with cf.ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(read, c): i for i, c in enumerate(chosen)}
        for f in cf.as_completed(futs):
            r = f.result()
            if r["text"].strip():
                read_ok[futs[f]] = r
                log(f"   ✔ {r['kind']}: {r['title'][:70]} ({len(r['text']):,} chars)")
    sources = [read_ok[i] for i in sorted(read_ok)] + extra
    for i, s in enumerate(sources, 1):
        s["id"] = i

    lines = [f"S{s['id']} · {s['kind']} · {s['title'][:80]} — {s['url'] or 'n/a'}" for s in sources]
    for n in notes:
        lines.append(f"ℹ️ {n}")
    if not sources:
        lines.append("No readable sources were found.")
    return "\n".join(lines), sources


def fmt_sources(sources: list[dict]) -> str:
    cap = max(1500, min(4500, EVIDENCE_CHARS // max(1, len(sources))))
    blocks = []
    for s in sources:
        limit = cap + 1000 if s["kind"] == "reddit" else cap
        blocks.append(
            f"[S{s['id']}] ({s['kind']}) {s['title']}\n{s['url']}\n{s['text'][:limit]}"
        )
    return "\n\n=====\n\n".join(blocks)


# ════════════════════════════════════════════════════════════════════════════
# 4. AGENT RUNNER (persona injection + search-on-demand)
# ════════════════════════════════════════════════════════════════════════════

SEARCH_RE = re.compile(r"\[\[\s*SEARCH\s*:\s*(.+?)\s*\]\]", re.I)

SEARCH_PROTOCOL = f"""
WEB SEARCH TOOL: If you need empirical grounding, current facts, or community/Reddit
verification before you can answer well, reply with ONLY up to {MAX_SEARCHES_PER_AGENT} lines of
the exact form
[[SEARCH: your query here]]
and nothing else. You will then receive the page text and must write your full answer.
For what real people think or experienced, put the word reddit in the query (it searches
Reddit threads and reads comments with their upvote scores).
Only search when it materially improves accuracy; otherwise answer directly.
"""


@dataclass
class AgentResult:
    text: str
    searches: list[str] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)


def build_system(cfg: Config, role_prompt: str, search_ok: bool) -> str:
    parts = []
    if cfg.instructions.strip():
        parts.append(f"USER SYSTEM INSTRUCTIONS (always follow):\n{cfg.instructions.strip()}")
    if cfg.personalization.strip():
        parts.append(f"USER PERSONALIZATION / CONTEXT:\n{cfg.personalization.strip()}")
    if cfg.evidence.strip():
        parts.append(
            "REAL-WORLD EVIDENCE BRIEF (gathered live from Reddit, forums and the web; cite as "
            "[S#]; never invent sources; if your reasoning conflicts with this evidence, say so "
            "and explain why):\n" + clip(cfg.evidence, 16000)
        )
    parts.append(role_prompt.strip())
    if search_ok:
        parts.append(SEARCH_PROTOCOL.strip())
    return "\n\n".join(parts)


def run_agent(
    rot: KeyRotator,
    cfg: Config,
    role_prompt: str,
    user: str,
    temperature: float = 0.6,
    allow_search: bool = False,
    buf: LiveBuffer | None = None,
) -> AgentResult:
    """Run one sub-agent (streamed); optionally let it trigger web searches mid-flight."""
    usage = Usage()
    search_ok = allow_search and cfg.search_enabled
    system = build_system(cfg, role_prompt, search_ok)
    text = call_llm(rot, cfg.model, system, user, temperature, buf=buf, usage=usage)
    searches: list[str] = []

    if search_ok:
        queries = SEARCH_RE.findall(text)[:MAX_SEARCHES_PER_AGENT]
        residue = SEARCH_RE.sub("", text).strip()
        if queries and len(residue) < 400:  # agent asked for search instead of answering
            if buf:
                buf.set("🔎 Searching the web: " + " | ".join(queries) + " …\n\n")
            evidence = []
            for q in queries:
                searches.append(q)
                evidence.append(f"### Query: {q}\n{search_web(q, cfg)}")
            if buf:
                buf.set("🔎 Searched: " + " | ".join(queries) + "\n\n")
            followup = (
                f"{user}\n\n=== LIVE WEB EVIDENCE ===\n"
                + "\n\n".join(evidence)
                + "\n=== END EVIDENCE ===\n\n"
                "Search is now closed. Write your COMPLETE answer, grounding claims in the "
                "evidence where relevant and citing source URLs inline. Do not emit "
                "[[SEARCH: ...]] again."
            )
            text = call_llm(
                rot, cfg.model, system, followup, temperature, buf=buf, usage=usage
            )
        text = SEARCH_RE.sub("", text).strip()
    return AgentResult(text=text, searches=searches, usage=usage)


# ════════════════════════════════════════════════════════════════════════════
# 5. STOPWATCH + RUN CONTEXT (all st.* calls stay on the main thread)
# ════════════════════════════════════════════════════════════════════════════

def fmt_elapsed(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s"


def clip(text: str, n: int = CONTEXT_CLIP) -> str:
    return text if len(text) <= n else text[:n] + "\n…[truncated]"


class Stopwatch:
    def __init__(self, placeholder):
        self.t0 = time.monotonic()
        self.ph = placeholder
        self.render()

    def elapsed(self) -> float:
        return time.monotonic() - self.t0

    def render(self) -> None:
        self.ph.markdown(f"⏱️ **Elapsed:** `{fmt_elapsed(self.elapsed())}`")


@dataclass
class Panel:
    """One agent's live area inside the status box: a header line + streaming text."""
    label: str
    hdr: object
    body: object
    buf: LiveBuffer
    shown: int = -1


@dataclass
class Run:
    cfg: Config
    rot: KeyRotator
    pool: cf.ThreadPoolExecutor
    clock: Stopwatch
    status: object
    problem: str
    token_cb: object  # callable(Usage) -> None, refreshes the sidebar token bar
    usage: Usage = field(default_factory=Usage)
    log: list[str] = field(default_factory=list)

    # -- UI helpers (main thread only) ---------------------------------------
    def begin(self, label: str) -> None:
        self.status.update(label=f"{label} · {fmt_elapsed(self.clock.elapsed())}")

    def panel(self, label: str) -> Panel:
        hdr, body = st.empty(), st.empty()
        hdr.markdown(f"▶️ **{label}** …")
        return Panel(label, hdr, body, LiveBuffer())

    def pump(self, futs: dict[cf.Future, Panel]):
        """Stream live text into the panels and tick the clock until all futures finish."""
        pending = set(futs)
        while pending:
            done, pending = cf.wait(pending, timeout=0.35, return_when=cf.FIRST_COMPLETED)
            self.clock.render()
            for f in pending:
                p = futs[f]
                v = p.buf.version
                if v != p.shown:
                    p.shown = v
                    p.body.markdown(p.buf.snapshot() + " ▌")
            for f in done:
                yield f

    def finish(self, p: Panel, res: AgentResult) -> None:
        self.usage.merge(res.usage)
        self.token_cb(self.usage)
        toks = f" · {res.usage.total:,} tokens" if res.usage.calls else ""
        p.hdr.markdown(f"✅ **{p.label}** — {fmt_elapsed(self.clock.elapsed())}{toks}")
        note = f"🔎 *Searched: {' | '.join(res.searches)}*\n\n" if res.searches else ""
        p.body.markdown(note + res.text)
        entry = f"## {p.label}\n\n"
        if res.searches:
            entry += f"*Web searches: {'; '.join(res.searches)}*\n\n"
        self.log.append(entry + res.text + "\n")
        self.clock.render()

    # -- execution ------------------------------------------------------------
    def task(self, label: str, fn):
        """A non-LLM job (searching and reading the web) that streams progress lines live."""
        self.begin(label)
        p = self.panel(label)
        fut = self.pool.submit(fn, p.buf)
        out = None
        for f in self.pump({fut: p}):
            out = f.result()
        log_text, data = out
        self.finish(p, AgentResult(text=log_text))
        return data

    def step(
        self, label: str, role: str, user: str,
        temperature: float = 0.6, allow_search: bool = False,
    ) -> str:
        """One sequential agent call, streamed live."""
        self.begin(label)
        p = self.panel(label)
        fut = self.pool.submit(
            run_agent, self.rot, self.cfg, role, user, temperature, allow_search, p.buf
        )
        res = None
        for f in self.pump({fut: p}):
            res = f.result()  # raises PipelineError if the call failed
        self.finish(p, res)
        return res.text

    def parallel(self, group_label: str, specs: dict[str, dict]) -> dict[str, str]:
        """Run several agents concurrently, each streaming into its own panel."""
        self.begin(group_label)
        st.markdown(f"**{group_label}**")
        panels = {k: self.panel(s["label"]) for k, s in specs.items()}
        futs = {
            self.pool.submit(
                run_agent, self.rot, self.cfg, s["role"], s["user"],
                s.get("temperature", 0.6), s.get("allow_search", False), panels[k].buf,
            ): k
            for k, s in specs.items()
        }
        out: dict[str, str] = {}
        for f in self.pump({f: panels[k] for f, k in futs.items()}):
            key = futs[f]
            res = f.result()
            self.finish(panels[key], res)
            out[key] = res.text
        # keep the log in declared order, not completion order
        order = [specs[k]["label"] for k in specs]
        tail = self.log[-len(specs):]
        tail.sort(key=lambda e: order.index(e.split("\n", 1)[0][3:]))
        self.log[-len(specs):] = tail
        return out


# ════════════════════════════════════════════════════════════════════════════
# 6. STAGE 1 — DEEP THINKING (FIRST PRINCIPLES)
# ════════════════════════════════════════════════════════════════════════════

def stage1_deep_thinking(run: Run) -> dict[str, str]:
    P = run.problem
    out: dict[str, str] = {}

    out["decon"] = run.step(
        "Stage 1 · Phase 1 — Problem Deconstruction",
        "You are a first-principles analyst. Strip the problem to bedrock: list every "
        "assumption hidden in the way it was stated and mark each as VALID / QUESTIONABLE / "
        "FALSE; identify the raw physical, economic, human and time constraints that are "
        "truly immovable; restate the problem in its most fundamental form; name what "
        "success would concretely look like and how it would be measured.",
        f"PROBLEM:\n{P}",
        temperature=0.3, allow_search=True,
    )

    out["diverge"] = run.step(
        "Stage 1 · Phase 2 — Lateral Divergence",
        "You are a lateral-thinking engine. Using ONLY the deconstruction provided, generate "
        "5-7 radically different approaches. They must differ in mechanism, not wording: "
        "no two may share the same underlying principle. Include at least one absurd-sounding "
        "idea and one that removes the problem instead of solving it. For each: a name, the "
        "core mechanism in 2-3 sentences, and why it might work.",
        f"PROBLEM:\n{P}\n\nDECONSTRUCTION:\n{clip(out['decon'])}",
        temperature=1.3,
    )

    out["redteam"] = run.step(
        "Stage 1 · Phase 3 — Inversion & Red-Teaming",
        "You are a ruthless red-team. Your mandate is to KILL every idea. For each one, invert "
        "it: assume it was implemented and failed spectacularly - write the post-mortem. "
        "List failure modes, hidden costs (money, time, attention, social friction, "
        "maintenance, second-order effects) and the assumption it silently depends on. "
        "Default verdict is KILLED; an idea earns SURVIVED or SURVIVED-WITH-AMENDMENT only "
        "if you genuinely cannot break it. Name exactly which component of each survivor "
        "held up. Check claims against the real world when you can.",
        f"PROBLEM:\n{P}\n\nCONSTRAINTS:\n{clip(out['decon'])}\n\nIDEAS:\n{clip(out['diverge'])}",
        temperature=0.4, allow_search=True,
    )

    out["steelman"] = run.step(
        "Stage 1 · Phase 4 — Steel-Manning",
        "You are a solution architect. Assemble ONE coherent solution STRICTLY from components "
        "that survived the red-team. List each component with the idea it came from; do not "
        "introduce new components. Apply the red-team's amendments. If nothing fully survived, "
        "say so plainly and build from the least-damaged parts with explicit repairs. Then "
        "present the strongest honest version of the solution, and state what it still cannot do.",
        f"PROBLEM:\n{P}\n\nCONSTRAINTS:\n{clip(out['decon'])}\n\nRED-TEAM VERDICTS:\n{clip(out['redteam'])}",
        temperature=0.4,
    )
    return out


# ════════════════════════════════════════════════════════════════════════════
# 7. STAGE 2 — RADICAL IDEATION (BASELINE BURNER)
# ════════════════════════════════════════════════════════════════════════════

def stage2_radical_ideation(run: Run, s1: dict[str, str]) -> dict[str, str]:
    P = run.problem
    out: dict[str, str] = {}

    out["burn"] = run.step(
        "Stage 2 · Step 1 — Baseline Burn",
        "You are the Baseline Burner. Identify the 5 most obvious, cliché solutions anyone "
        "would propose first (the ones a search engine or a generic AI would give). Number "
        "them, state each in one line, and then explicitly BAN them: write 'BANNED:' followed "
        "by the list. Also list the lazy patterns behind them (e.g. 'add an app', "
        "'send reminders') that are banned too.",
        f"PROBLEM:\n{P}",
        temperature=0.5,
    )

    out["transpose"] = run.step(
        "Stage 2 · Step 2 — Cross-Disciplinary Transposition",
        "You are a cross-disciplinary inventor. Borrow mechanisms EXCLUSIVELY from these four "
        "domains: (1) biomimicry, (2) game psychology, (3) naval/military logistics, "
        "(4) manufacturing. Produce at least one transposed concept per domain (6-8 total). "
        "For each: the source mechanism, how it works in its home domain, the exact mapping "
        "onto this problem, and what it would look like in practice. No concept may resemble "
        "anything on the banned list - confirm this at the end.",
        f"PROBLEM:\n{P}\n\nRAW CONSTRAINTS (from first-principles analysis):\n{clip(s1['decon'])}"
        f"\n\nBANNED CLICHÉS:\n{clip(out['burn'])}",
        temperature=1.3,
    )

    out["invert"] = run.step(
        "Stage 2 · Step 3 — Constraint Inversion",
        "You are a stress-tester. Pick the single best concept from the list and justify the "
        "pick. Then stress-test it under extreme constraints: ZERO software and ZERO screens "
        "(purely physical and social), zero budget, one person, ten minutes per day, hostile "
        "environment. Show what breaks at each extreme and redesign the concept so the "
        "core mechanism survives. State the irreducible core: what is the one thing that "
        "makes it work?",
        f"PROBLEM:\n{P}\n\nCONCEPTS:\n{clip(out['transpose'])}",
        temperature=0.7,
    )

    out["practical"] = run.step(
        "Stage 2 · Step 4 — Practical Extraction",
        "You are an operator. Ground the core mechanism into a feasible real-world plan: "
        "exact materials/resources with rough costs, step-by-step setup, the first 48 hours, "
        "a 30-day timeline, owner of each action, measurable success metrics, leading "
        "indicators, and tripwires that tell you to stop or adapt. Be concrete enough that "
        "someone could start today. Verify any real-world facts you rely on.",
        f"PROBLEM:\n{P}\n\nSTRESS-TESTED CONCEPT:\n{clip(out['invert'])}",
        temperature=0.4, allow_search=True,
    )
    return out


# ════════════════════════════════════════════════════════════════════════════
# 8. STAGE 3 — THE TRIAD DEBATE
# ════════════════════════════════════════════════════════════════════════════

def stage3_triad_debate(run: Run, s1: dict[str, str], s2: dict[str, str]) -> dict[str, str]:
    P = run.problem
    proposals = (
        f"PROPOSAL A (first-principles, steel-manned):\n{clip(s1['steelman'])}\n\n"
        f"PROPOSAL B (radical ideation, practical plan):\n{clip(s2['practical'])}"
    )
    base = f"PROBLEM:\n{P}\n\n{proposals}"

    triad = run.parallel(
        "Stage 3 — The Triad Debate (3 agents in parallel)",
        {
            "proponent": dict(
                label="Stage 3 · The Proponent (Builder)",
                role="You are The Proponent. Build the strongest possible case and the strongest "
                     "possible plan. Merge the best of Proposal A and B into one executable "
                     "design, fill gaps, answer obvious objections in advance, and state "
                     "exactly what must be true for it to succeed.",
                user=base, temperature=0.6,
            ),
            "inquisitor": dict(
                label="Stage 3 · The Inquisitor (Falsifier)",
                role="You are The Inquisitor. Your job is falsification. For every key claim in "
                     "the proposals, ask: what evidence would prove this wrong, and does that "
                     "evidence exist? Demand proof, expose untested assumptions, fabricated-"
                     "looking facts, survivorship bias and missing numbers. Look up empirical "
                     "evidence or real user reports (including Reddit) when it can confirm or "
                     "refute a claim. End with the 3 claims most likely to be false.",
                user=base, temperature=0.4, allow_search=True,
            ),
            "contrarian": dict(
                label="Stage 3 · The Contrarian (Premise Attacker)",
                role="You are The Contrarian. Do NOT critique the execution; attack the PREMISE. "
                     "Is the problem framed correctly? Is it even the right problem? Is the "
                     "stated goal the real goal? What if the opposite approach (do less, do "
                     "nothing, remove the thing) is better? Offer at least one reframing that "
                     "would make both proposals irrelevant, and say when you would be right.",
                user=base, temperature=0.9, allow_search=True,
            ),
        },
    )

    verdict = run.step(
        "Stage 3 · The Arbiter (Judge)",
        "You are The Arbiter, an impartial judge. You have heard the Proponent, the "
        "Inquisitor and the Contrarian. Resolve the friction between them: for each major "
        "dispute, state who is right and why; decide which attacks land and which are "
        "overblown; decide whether the Contrarian's reframing changes the plan. Deliver a "
        "VERDICT: the final recommended plan (amended where attacks landed), the conditions "
        "under which it holds, unresolved risks, a confidence level (low/medium/high) with "
        "reasons, and what evidence would change your mind. Do not hedge for its own sake.",
        f"{base}\n\n=== THE PROPONENT ===\n{clip(triad['proponent'])}"
        f"\n\n=== THE INQUISITOR ===\n{clip(triad['inquisitor'])}"
        f"\n\n=== THE CONTRARIAN ===\n{clip(triad['contrarian'])}",
        temperature=0.4,
    )
    triad["verdict"] = verdict
    return triad


# ════════════════════════════════════════════════════════════════════════════
# 9. STAGE 4 — REVIEW & SYNTHESIS
# ════════════════════════════════════════════════════════════════════════════

def stage4_review_and_synthesis(run: Run, s3: dict[str, str]) -> str:
    P = run.problem
    verdict = clip(s3["verdict"])
    base = f"PROBLEM:\n{P}\n\nARBITER'S VERDICT:\n{verdict}"

    reviews = run.parallel(
        "Stage 4 — Review (2 reviewers in parallel)",
        {
            "logic": dict(
                label="Stage 4 · Reviewer 1 (Logic Audit)",
                role="You are Reviewer 1, a logic auditor. Audit the verdict for logical gaps: "
                     "non-sequiturs, circular reasoning, contradictions, unstated assumptions, "
                     "steps that don't follow, missing dependencies, and conclusions stronger "
                     "than their support. For each issue: quote the location, explain the flaw, "
                     "and give the fix. If the logic is sound somewhere, say so briefly.",
                user=base, temperature=0.2,
            ),
            "facts": dict(
                label="Stage 4 · Reviewer 2 (Hallucination Check)",
                role="You are Reviewer 2, a hallucination and fact auditor. Flag every statistic, "
                     "named study, source, product, price, law, or empirical claim in the "
                     "verdict. Classify each as VERIFIED / PLAUSIBLE-UNVERIFIED / LIKELY-"
                     "FABRICATED, verifying with a web search when it is checkable. Output a "
                     "list of claims to remove or soften and the corrected wording.",
                user=base, temperature=0.2, allow_search=True,
            ),
        },
    )

    final = run.step(
        "Stage 4 · The Final Writer",
        "You are the Final Writer. Produce the decisive final report for the person who asked. "
        "Start from the Arbiter's verdict, apply every valid correction from both reviewers, "
        "and delete or soften anything flagged as fabricated or unverified. Be direct and "
        "commit to a recommendation. Structure: **Bottom line** (2-3 sentences), **Why this "
        "wins**, **The plan** (concrete steps and timeline), **Risks & tripwires**, **What we "
        "rejected and why**, **Confidence & what would change it**. Use clear markdown, no "
        "filler, and do not mention internal stages, agents or the review process. If an evidence "
        "brief with [S#] sources is available, keep [S#] citations on the key claims, make clear "
        "what real people reported versus what is your reasoning, and end with a **Sources** "
        "list (title and URL for each source you cited).",
        f"{base}\n\n=== REVIEWER 1 (LOGIC) ===\n{clip(reviews['logic'])}"
        f"\n\n=== REVIEWER 2 (FACTS) ===\n{clip(reviews['facts'])}",
        temperature=0.5,
    )
    return final


# ════════════════════════════════════════════════════════════════════════════
# 10. ORCHESTRATION
# ════════════════════════════════════════════════════════════════════════════

PLANNER_ROLE = (
    "You are a research planner. The team will solve the problem below. Plan LIVE web research "
    "that surfaces what REAL PEOPLE report from actual experience, not marketing or SEO "
    "listicles. Output exactly 8 lines and nothing else, each of the form `KIND: search query`:\n"
    "- 4 lines REDDIT: queries phrased the way people write on Reddit (first person, specific; "
    "name a likely subreddit like r/xyz when you know one; angles like 'worked for me', "
    "'regret', 'tried for a year', 'alternatives').\n"
    "- 2 lines COMMUNITY: forums, Hacker News, StackExchange, specialist communities, review threads.\n"
    "- 2 lines WEB: authoritative or primary sources, data, studies, official docs.\n"
    "Keep each query under 12 words. Cover different angles: what works, what fails, hidden "
    "costs and problems, and alternatives."
)

FIELD_ROLE = (
    "You are the Field Researcher. You are handed raw material: Reddit threads (with comment "
    "upvote scores), forum threads and web pages, each labelled [S#]. Write an EVIDENCE BRIEF "
    "telling the team what is actually true in the real world.\n"
    "Rules: use ONLY the supplied material; never invent sources, quotes or numbers; cite every "
    "claim with [S#]. Weight first-hand experience and high-upvote comments above marketing, "
    "SEO listicles and self-promotion, and say when a claim rests on a single person.\n"
    "Sections: 1) What real people report (consensus, and how many sources agree); "
    "2) What worked, what failed, regrets and hidden costs; 3) Disagreements and minority views; "
    "4) Practical specifics people recommend (tools, numbers, steps); 5) Credibility warnings "
    "(old threads, small samples, promotion, contradictions); 6) Gaps the sources did NOT answer.\n"
    "Be dense and specific, paraphrase instead of long quotes, and do not write the final plan, "
    "only the evidence."
)


def parse_plan(plan: str, problem: str) -> list[tuple[str, str]]:
    qs: list[tuple[str, str]] = []
    for m in re.finditer(
        r"^[\s\-\*\d\.\)]*(REDDIT|COMMUNITY|WEB)\s*:\s*(.+?)\s*$", plan, re.I | re.M
    ):
        q = m.group(2).strip().strip("`\"'*")
        if q:
            qs.append((m.group(1).upper(), q[:160]))
    if not qs:  # planner ignored the format: fall back to the raw problem
        p = re.sub(r"\s+", " ", problem).strip()[:140]
        qs = [("REDDIT", p), ("REDDIT", p + " worked for me"), ("COMMUNITY", p + " forum"), ("WEB", p)]
    return qs[:10]


def stage0_research(run: Run) -> str:
    P = run.problem
    plan = run.step(
        "Stage 0 · Step 1 — Research Plan", PLANNER_ROLE, f"PROBLEM:\n{P}", temperature=0.4
    )
    queries = parse_plan(plan, P)
    sources = run.task(
        "Stage 0 · Step 2 — Gathering Sources (Reddit threads, forums, web)",
        lambda buf: gather_sources(run.cfg, run.rot, queries, buf),
    )
    if not sources:
        return NO_EVIDENCE
    brief = run.step(
        "Stage 0 · Step 3 — Field Report",
        FIELD_ROLE,
        f"PROBLEM:\n{P}\n\nRAW SOURCE MATERIAL:\n{fmt_sources(sources)}",
        temperature=0.2,
    )
    listing = "\n".join(
        f"[S{s['id']}] ({s['kind']}) {s['title'][:90]} — {s['url'] or 'Google Search grounding'}"
        for s in sources
    )
    return f"{brief}\n\nSOURCES:\n{listing}"


def _stage(run: Run, ckpt: dict, name: str, fn):
    """Run a stage, or restore it from an earlier failed attempt (costs no API calls)."""
    if name in ckpt["done"]:
        run.log.extend(ckpt["logs"][name])
        st.markdown(f"♻️ **{name}** restored from the previous attempt — no API calls used")
        return ckpt["done"][name]
    start = len(run.log)
    out = fn()
    ckpt["done"][name] = out
    ckpt["logs"][name] = run.log[start:]
    return out


def execute_pipeline(run: Run, ckpt: dict) -> str:
    if run.cfg.research_enabled:
        brief = _stage(run, ckpt, "Stage 0", lambda: stage0_research(run))
        if brief == NO_EVIDENCE:  # don't keep a failed research result for the retry
            ckpt["done"].pop("Stage 0", None)
            ckpt["logs"].pop("Stage 0", None)
        run.cfg.evidence = brief
    s1 = _stage(run, ckpt, "Stage 1", lambda: stage1_deep_thinking(run))
    s2 = _stage(run, ckpt, "Stage 2", lambda: stage2_radical_ideation(run, s1))
    s3 = _stage(run, ckpt, "Stage 3", lambda: stage3_triad_debate(run, s1, s2))
    return _stage(run, ckpt, "Stage 4", lambda: stage4_review_and_synthesis(run, s3))


def build_full_log(
    cfg: Config, problem: str, run_log: list[str], elapsed: float, final: str, tokens: int
) -> str:
    head = [
        "# Full Thinking / Deliberation Log",
        f"*Model: {cfg.model} · Total time: {fmt_elapsed(elapsed)} · Tokens: {tokens:,}*",
        "",
        "## Problem",
        problem,
    ]
    if cfg.instructions.strip():
        head += ["", "## System Instructions", cfg.instructions.strip()]
    if cfg.personalization.strip():
        head += ["", "## Personalization", cfg.personalization.strip()]
    body = "\n\n".join(run_log)
    tail = f"\n\n## FINAL OUTPUT\n\n{final}" if final else ""
    return "\n".join(head) + "\n\n---\n\n" + body + tail


# ════════════════════════════════════════════════════════════════════════════
# 11. UI  (sidebar, token bar, saved chats, results)
# ════════════════════════════════════════════════════════════════════════════

def init_state() -> None:
    ss = st.session_state
    ss.setdefault("history", [])       # saved chats (this browser session)
    ss.setdefault("tokens_total", 0)   # tokens used this session
    ss.setdefault("result", None)      # chat currently shown
    ss.setdefault("problem_input", "")


def _load_chat(chat_id: str) -> None:
    for c in st.session_state.history:
        if c["id"] == chat_id:
            st.session_state.result = c
            st.session_state.problem_input = c["problem"]
            return


def _new_chat() -> None:
    st.session_state.result = None
    st.session_state.problem_input = ""


def render_token_panel(box, run_usage: Usage | None, base_total: int, budget: int) -> None:
    used = base_total + (run_usage.total if run_usage else 0)
    frac = min(1.0, used / budget) if budget else 0.0
    with box.container():
        st.progress(frac, text=f"Session tokens: {used:,} / {budget:,} ({frac:.0%})")
        if run_usage is not None:
            approx = " (estimated)" if run_usage.estimated else ""
            st.caption(
                f"This run: {run_usage.prompt:,} in · {run_usage.completion:,} out · "
                f"{run_usage.calls} calls{approx}"
            )


def render_history(box, tag: str) -> None:
    """tag keeps widget keys unique when the list is drawn twice in one script run."""
    with box.container():
        hist = st.session_state.history
        if not hist:
            st.caption("No saved chats yet.")
        for c in reversed(hist[-30:]):
            st.button(
                f"💬 {c['title']}", key=f"chat_{c['id']}_{tag}",
                on_click=_load_chat, args=(c["id"],),
            )


def import_chats(uploaded) -> None:
    sig = f"{uploaded.name}:{uploaded.size}"
    if st.session_state.get("_imported") == sig:
        return
    try:
        data = json.loads(uploaded.getvalue().decode("utf-8"))
        have = {c["id"] for c in st.session_state.history}
        need = {"id", "title", "problem", "final", "log"}
        added = 0
        for c in data:
            if isinstance(c, dict) and need <= set(c) and c["id"] not in have:
                st.session_state.history.append(c)
                added += 1
        st.session_state["_imported"] = sig
        st.success(f"Imported {added} chat(s).")
    except Exception as e:
        st.error(f"Could not read that file: {e}")


def sidebar():
    with st.sidebar:
        st.header("⚙️ Settings")
        instructions = st.text_area(
            "System Instructions",
            height=140,
            placeholder="Rules every agent must follow, e.g. 'Be blunt. No corporate jargon.'",
            key="sys_instr",
        )
        personalization = st.text_area(
            "Personalization",
            height=140,
            placeholder="Who you are, your context, constraints, preferences…",
            key="personalization",
        )
        st.caption("Both are injected into every agent in every stage.")
        model = st.selectbox("Model", MODEL_CHOICES, index=0)
        custom = st.text_input(
            "Custom model name (optional)",
            placeholder="overrides the dropdown, e.g. gemini-3.8-flash",
        ).strip()
        if custom:
            model = custom
        rpm = st.number_input(
            "Requests per minute, per key", min_value=1, max_value=60, value=RPM_PER_KEY,
            help="Check your real limit at aistudio.google.com/rate-limit",
        )
        search_enabled = st.toggle("Allow dynamic web search", value=True)
        research_enabled = st.toggle(
            "Deep research stage (Reddit, forums, web)", value=True,
            help="Runs first: plans searches, reads real Reddit threads and forums, and feeds "
                 "the evidence to every stage. Costs about 2 extra AI calls.",
        )
        use_grounding = st.toggle(
            "Use Google Search grounding", value=False,
            help="Google's own search (what AI Studio uses). Free API keys usually can't use it; "
                 "it needs a billing-enabled project. If unavailable, the app falls back automatically.",
        )
        tavily_key = _secret("TAVILY_API_KEY")
        jina_key = _secret("JINA_API_KEY")
        if tavily_key:
            st.success("Tavily search key found")
        else:
            st.info("No TAVILY_API_KEY, so DuckDuckGo is used (weaker). A free Tavily key makes search much better.")
        budget = st.number_input(
            "Token budget (your own limit)", min_value=10_000, value=500_000, step=50_000,
            help="Only drives the bar below. Google's real quota is at aistudio.google.com/rate-limit",
        )
        n_keys = len(load_keys())
        if n_keys:
            st.success(f"{n_keys} API key(s) loaded · rotating every call")
        else:
            st.error("No API keys found")

        st.divider()
        st.subheader("📊 Tokens")
        token_box = st.empty()

        st.divider()
        st.subheader("💬 Saved chats")
        st.button("➕ New chat", on_click=_new_chat)
        hist_box = st.empty()
        st.caption(
            "Chats live only while this browser tab session lasts. "
            "Export to keep them, import to bring them back."
        )
        st.download_button(
            "⬇️ Export chats (.json)",
            data=json.dumps(st.session_state.history, indent=2),
            file_name="chats.json",
            mime="application/json",
        )
        up = st.file_uploader("Import chats", type="json")
        if up is not None:
            import_chats(up)

    cfg = Config(
        model, instructions, personalization, search_enabled, int(rpm), int(budget),
        tavily_key=tavily_key, jina_key=jina_key,
        research_enabled=research_enabled, use_grounding=use_grounding,
    )
    return cfg, token_box, hist_box


def copy_box(label: str, text: str, height: int = 320) -> None:
    """st.code ships a built-in one-click copy button (top-right of the block)."""
    st.markdown(f"**{label}**")
    for kwargs in (
        dict(wrap_lines=True, height=height),
        dict(wrap_lines=True),
        dict(),
    ):
        try:
            st.code(text, language=None, **kwargs)
            return
        except TypeError:  # older Streamlit without these args
            continue


def render_result(result: dict) -> None:
    if result.get("error"):
        st.error(result["error"])
    st.markdown(
        f"⏱️ **Total time:** `{fmt_elapsed(result['elapsed'])}` · "
        f"🔢 **Tokens:** `{result.get('tokens', 0):,}` · "
        f"🤖 `{result.get('model', '?')}`"
    )
    if result["final"]:
        st.subheader("📄 Final Report")
        st.markdown(result["final"])
    st.divider()
    st.subheader("📋 Copy boxes")
    if result["final"]:
        copy_box("Final Output", result["final"])
    copy_box("Full Thinking / Deliberation Log", result["log"], height=420)


def main() -> None:
    st.set_page_config(page_title="Cognitive Pipeline", page_icon="🧠", layout="wide")
    init_state()
    st.title("🧠 Multi-Stage Cognitive Pipeline")
    st.caption("Deep Thinking → Radical Ideation → Triad Debate → Review & Synthesis")

    cfg, token_box, hist_box = sidebar()
    base_tokens = st.session_state.tokens_total
    render_token_panel(token_box, None, base_tokens, cfg.token_budget)
    render_history(hist_box, "a")

    problem = st.text_area(
        "Your problem or question",
        height=150,
        key="problem_input",
        placeholder="Describe the problem you want solved…",
    )
    sig = repr((problem.strip(), cfg.instructions, cfg.personalization))
    ckpt = st.session_state.get("ckpt")
    if not ckpt or ckpt["sig"] != sig:
        ckpt = {"sig": sig, "done": {}, "logs": {}}
        st.session_state.ckpt = ckpt
    if ckpt["done"]:
        st.info(
            "♻️ Progress from the last attempt is saved (" + ", ".join(ckpt["done"]) +
            "). Click Run again to continue from where it stopped."
        )
    go = st.button("🚀 Run pipeline", type="primary", disabled=not problem.strip())

    if go:
        keys = load_key_items()
        if not keys:
            st.error(
                "Add API_KEY_1, API_KEY_2, API_KEY_3 to .streamlit/secrets.toml "
                "or your environment variables."
            )
            st.stop()

        rot = get_rotator(tuple(keys), cfg.rpm, ROTATOR_VERSION)
        timer_ph = st.empty()
        clock = Stopwatch(timer_ph)
        error = None
        final = ""
        usage = Usage()
        run_log: list[str] = []

        with cf.ThreadPoolExecutor(max_workers=4) as pool:
            with st.status("🧠 Starting pipeline…", expanded=True) as status:
                run = Run(
                    cfg, rot, pool, clock, status, problem.strip(),
                    token_cb=lambda u: render_token_panel(
                        token_box, u, base_tokens, cfg.token_budget
                    ),
                )
                try:
                    final = execute_pipeline(run, ckpt)
                    st.session_state.ckpt = {"sig": sig, "done": {}, "logs": {}}
                    status.update(
                        label=f"✅ Pipeline complete in {fmt_elapsed(clock.elapsed())}",
                        state="complete", expanded=False,
                    )
                except PipelineError as e:
                    error = f"Pipeline stopped: {e}"
                    status.update(
                        label=f"❌ Failed after {fmt_elapsed(clock.elapsed())}",
                        state="error", expanded=True,
                    )
                except Exception as e:  # never lose the partial log
                    error = f"Unexpected error: {type(e).__name__}: {e}"
                    status.update(label="❌ Unexpected error", state="error", expanded=True)
                usage, run_log = run.usage, run.log

        elapsed = clock.elapsed()
        timer_ph.empty()
        if rot.bad:
            st.warning(
                "Google rejected these keys as invalid, so they were skipped: "
                + ", ".join(sorted(rot.bad))
                + ". Re-copy them from aistudio.google.com/apikey and fix them in Secrets."
            )
        st.session_state.tokens_total += usage.total
        render_token_panel(token_box, usage, base_tokens, cfg.token_budget)

        title = problem.strip().replace("\n", " ")
        title = title[:45] + ("…" if len(title) > 45 else "")
        entry = {
            "id": uuid.uuid4().hex[:10],
            "title": title,
            "problem": problem.strip(),
            "final": final,
            "log": build_full_log(cfg, problem.strip(), run_log, elapsed, final, usage.total),
            "elapsed": elapsed,
            "tokens": usage.total,
            "model": cfg.model,
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "error": error,
        }
        st.session_state.result = entry
        if final:
            st.session_state.history.append(entry)
            render_history(hist_box, "b")

    if st.session_state.result:
        render_result(st.session_state.result)


if __name__ == "__main__":
    main()
