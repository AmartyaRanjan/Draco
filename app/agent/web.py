"""Web access: SSRF-safe fetching, readable-text extraction, search providers.

Everything fetched is untrusted data. The fetcher validates scheme, credentials, port, resolves DNS
itself and rejects any non-public address, follows redirects manually (re-validating every hop),
connects to the validated IP (so DNS rebinding between check and connect cannot redirect it),
caps size/time and allows only text-like content types.
"""
import asyncio
import hashlib
import ipaddress
import re
import socket
import time
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx

ALLOWED_TYPES = ("text/html", "application/xhtml+xml", "text/plain", "text/markdown")
_INJECTION = re.compile(
    r"(ignore (all |any |the )?(previous|prior|above) (instructions|prompts)|disregard (the )?(system|previous)|"
    r"you are now|new instructions:|system prompt|<\|im_start\|>|</?(system|assistant)>|"
    r"(reveal|print|send|disclose) .{0,30}(api[ _-]?key|password|secret)|"
    r"disregard (what|the user|your|all|any)|(reply|respond|answer|output|write) (only |exactly )?with (exactly |only )?(?-i:[\"'`A-Z])|"
    r"do not (answer|follow|obey) (the )?(user|question|above)|call (the )?(tool|function|capability) )", re.I)


def looks_injected(text: str) -> bool:
    """Heuristic only: instruction-like text inside fetched content. Used to withhold passages from the model, as one
    layer among several (untrusted labelling, no approval-free side effects, citation validation). Paraphrases get past it."""
    return bool(_INJECTION.search(text))


class UrlPolicyError(Exception):
    def __init__(self, message, code="unsafe_url"):
        super().__init__(message)
        self.code = code


def normalize_url(url: str) -> str:
    p = urlsplit(url.strip())
    host = (p.hostname or "").lower().rstrip(".")
    port = f":{p.port}" if p.port and p.port not in (80, 443) else ""
    path = p.path or "/"
    return urlunsplit((p.scheme.lower(), host + port, path, p.query, ""))


def url_hash(url: str) -> str:
    return hashlib.sha256(normalize_url(url).encode()).hexdigest()


def check_url(url: str, allowed_ports=(80, 443)) -> urlsplit:
    if len(url) > 2048:
        raise UrlPolicyError("URL too long")
    try:
        p = urlsplit(url.strip())
        port = p.port
    except ValueError as exc:
        raise UrlPolicyError("Malformed URL") from exc
    if p.scheme not in ("http", "https"):
        raise UrlPolicyError("Only http and https URLs are allowed")
    if not p.hostname:
        raise UrlPolicyError("URL has no host")
    if p.username or p.password:
        raise UrlPolicyError("URLs with credentials are not allowed")
    if port is not None and port not in allowed_ports:
        raise UrlPolicyError(f"Port {port} is not allowed")
    return p


def is_public_ip(ip: str) -> bool:
    a = ipaddress.ip_address(ip)
    if isinstance(a, ipaddress.IPv6Address) and a.ipv4_mapped:
        a = a.ipv4_mapped
    return a.is_global and not a.is_multicast


async def default_resolver(host: str, port: int) -> list[str]:
    infos = await asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return list(dict.fromkeys(i[4][0] for i in infos))


@dataclass
class FetchedPage:
    requested_url: str
    final_url: str
    status_code: int
    content_type: str
    text: str            # decoded body (possibly truncated), still raw HTML for html types
    truncated: bool
    redirects: list[str] = field(default_factory=list)
    fetched_at: float = field(default_factory=time.time)


class WebFetcher:
    def __init__(self, settings, *, resolver=default_resolver, transport: httpx.AsyncBaseTransport | None = None):
        self.s, self.resolver, self.transport = settings, resolver, transport

    async def _validated_ip(self, p) -> str:
        port = p.port or (443 if p.scheme == "https" else 80)
        host = p.hostname
        try:
            ipaddress.ip_address(host)
            addrs = [host]
        except ValueError:
            try:
                addrs = await self.resolver(host, port)
            except OSError as exc:
                raise UrlPolicyError("Host could not be resolved", "upstream_error") from exc
        if not addrs:
            raise UrlPolicyError("Host could not be resolved", "upstream_error")
        if not self.s.web_allow_private_hosts and not all(is_public_ip(a) for a in addrs):
            raise UrlPolicyError("Host resolves to a non-public address")
        return addrs[0]

    async def fetch(self, url: str) -> FetchedPage:
        s, redirects, cur = self.s, [], url
        async with httpx.AsyncClient(transport=self.transport, follow_redirects=False, trust_env=False,
                                     timeout=httpx.Timeout(s.web_timeout_s), verify=True) as client:
            for _hop in range(s.web_max_redirects + 1):
                p = check_url(cur, s.web_allowed_ports)
                ip = await self._validated_ip(p)
                port = p.port or (443 if p.scheme == "https" else 80)
                netloc = f"[{ip}]" if ":" in ip else ip
                target = urlunsplit((p.scheme, f"{netloc}:{port}", p.path or "/", p.query, ""))
                headers = {"Host": p.hostname if not p.port else f"{p.hostname}:{p.port}",
                           "User-Agent": "rag-agent-fetcher/1.0", "Accept": "text/html,text/plain;q=0.9",
                           "Accept-Encoding": "gzip"}
                ext = {"sni_hostname": p.hostname} if p.scheme == "https" else {}
                async with client.stream("GET", target, headers=headers, extensions=ext) as r:
                    if r.status_code in (301, 302, 303, 307, 308) and r.headers.get("location"):
                        cur = urljoin(cur, r.headers["location"])
                        redirects.append(cur)
                        continue
                    ctype = r.headers.get("content-type", "").split(";")[0].strip().lower()
                    if r.status_code >= 400:
                        raise UrlPolicyError(f"Upstream returned HTTP {r.status_code}", "upstream_error")
                    if ctype not in ALLOWED_TYPES:
                        raise UrlPolicyError(f"Content type '{ctype or 'unknown'}' is not allowed", "unsafe_url")
                    buf, truncated = bytearray(), False
                    async for chunk in r.aiter_bytes():  # counts decoded bytes: bounds decompression bombs too
                        buf += chunk
                        if len(buf) >= s.web_max_bytes:
                            truncated = True
                            del buf[s.web_max_bytes:]
                            break
                    enc = r.encoding or "utf-8"
                    return FetchedPage(url, cur, r.status_code, ctype, bytes(buf).decode(enc, errors="replace"),
                                       truncated, redirects)
            raise UrlPolicyError("Too many redirects", "limit_exceeded")


# ---- extraction ---------------------------------------------------------------------------
@dataclass
class Passage:
    passage_id: str
    text: str
    start: int
    end: int


@dataclass
class ExtractedPage:
    title: str | None
    text: str
    passages: list[Passage]
    tables: list[list[list[str]]]
    injection_suspected: bool


def html_to_page(raw: str, content_type: str = "text/html", *, max_chars: int = 400_000,
                 max_passage_chars: int = 1200) -> ExtractedPage:
    title, tables = None, []
    if content_type in ("text/plain", "text/markdown"):
        text = raw
    else:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(raw, "lxml")
        title = (soup.title.string or "").strip() if soup.title and soup.title.string else None
        for t in soup(["script", "style", "noscript", "iframe", "svg", "form", "nav", "footer", "aside", "template"]):
            t.decompose()
        for tb in soup.find_all("table")[:10]:
            rows = [[c.get_text(" ", strip=True)[:200] for c in tr.find_all(["th", "td"])] for tr in tb.find_all("tr")[:60]]
            rows = [r for r in rows if any(r)]
            if rows:
                tables.append(rows)
        text = soup.get_text("\n")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text).strip()[:max_chars]
    return ExtractedPage(title, text, split_passages(text, max_passage_chars), tables,
                         bool(_INJECTION.search(text)))


def split_passages(text: str, max_chars: int = 1200, min_chars: int = 200) -> list[Passage]:
    out, buf, buf_start, pos = [], "", 0, 0
    for para in re.split(r"\n\n+", text):
        start = text.find(para, pos)
        pos = start + len(para)
        if not para.strip():
            continue
        pieces = [para]
        if len(para) > max_chars:  # split long paragraphs on sentence boundaries
            pieces, cur = [], ""
            for sent in re.split(r"(?<=[.!?])\s+", para):
                if cur and len(cur) + len(sent) + 1 > max_chars:
                    pieces.append(cur)
                    cur = ""
                cur = f"{cur} {sent}".strip()
                while len(cur) > max_chars:  # unbroken run: hard split
                    pieces.append(cur[:max_chars])
                    cur = cur[max_chars:]
            if cur:
                pieces.append(cur)
        for pc in pieces:
            if not buf:
                buf_start = start
            buf = f"{buf}\n{pc}".strip() if buf else pc
            if len(buf) >= min_chars:
                out.append(Passage(f"p{len(out) + 1}", buf, buf_start, buf_start + len(buf)))
                buf = ""
    if buf:
        out.append(Passage(f"p{len(out) + 1}", buf, buf_start, buf_start + len(buf)))
    return out


_W = re.compile(r"\w+", re.U)
_STOP = frozenset("the a an of to in on for and or is are was were be by with as at that this it from what which who how".split())


def rank_passages(query: str, passages: list[Passage], k: int) -> list[tuple[Passage, float]]:
    """Lexical relevance (idf-weighted term overlap). Deterministic; no model call."""
    q = {w.lower() for w in _W.findall(query)} - _STOP
    if not q or not passages:
        return []
    docs = [{w.lower() for w in _W.findall(p.text)} for p in passages]
    n = len(passages)
    idf = {t: 1 + (n / (1 + sum(t in d for d in docs))) for t in q}
    scored = [(p, sum(idf[t] for t in q & d) / sum(idf.values())) for p, d in zip(passages, docs)]
    return sorted((x for x in scored if x[1] > 0), key=lambda x: -x[1])[:k]


# ---- search providers ---------------------------------------------------------------------
@dataclass
class SearchHit:
    url: str
    title: str
    snippet: str
    date: str | None = None


class SearchProvider:
    name = "base"

    async def search(self, query: str, n: int) -> list[SearchHit]:
        raise NotImplementedError


class SearxngProvider(SearchProvider):
    """Self-hosted SearXNG with the JSON format enabled (GET {base}/search?q=...&format=json)."""
    name = "searxng"

    def __init__(self, base_url, timeout=10, transport=None):
        self.base, self.timeout, self.transport = base_url.rstrip("/"), timeout, transport

    async def search(self, query, n):
        async with httpx.AsyncClient(timeout=self.timeout, transport=self.transport) as c:
            r = await c.get(f"{self.base}/search", params={"q": query, "format": "json"})
            r.raise_for_status()
        return [SearchHit(x["url"], x.get("title", ""), x.get("content", ""), x.get("publishedDate"))
                for x in r.json().get("results", [])[:n] if x.get("url")]


class BraveProvider(SearchProvider):
    """Brave Search API (GET /res/v1/web/search with X-Subscription-Token)."""
    name = "brave"

    def __init__(self, api_key, timeout=10, transport=None, base="https://api.search.brave.com"):
        self.key, self.timeout, self.transport, self.base = api_key, timeout, transport, base

    async def search(self, query, n):
        async with httpx.AsyncClient(timeout=self.timeout, transport=self.transport) as c:
            r = await c.get(f"{self.base}/res/v1/web/search", params={"q": query, "count": n},
                            headers={"X-Subscription-Token": self.key, "Accept": "application/json"})
            r.raise_for_status()
        return [SearchHit(x["url"], x.get("title", ""), x.get("description", ""), x.get("age") or x.get("page_age"))
                for x in r.json().get("web", {}).get("results", [])[:n] if x.get("url")]


def build_search_provider(s) -> SearchProvider | None:
    if s.web_search_provider == "searxng" and s.web_search_url:
        return SearxngProvider(s.web_search_url, s.web_timeout_s)
    if s.web_search_provider == "brave" and s.web_search_api_key:
        return BraveProvider(s.web_search_api_key, s.web_timeout_s)
    return None
