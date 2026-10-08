"""
AILYZE GetData — ChatGPT App (Python MCP server)

Polling-only flow:
1) User asks for export.
2) Model calls tool download_data(urls, top_n).
3) Server creates ExportComments jobs (POST /api/v3/job), returns structuredContent + widget template.
4) Widget polls via MCP tool check_job_status (GET /api/v3/job/{guid}).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests
from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, TextContent

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
BASE_URL = os.getenv("EXPORTCOMMENTS_BASE_URL", "https://exportcomments.com").rstrip("/")
API_KEY = os.getenv("EXPORT_API_TOKEN", "").strip()



CREATE_TIMEOUT: Tuple[int, int] = (8, 45)   # connect, read
STATUS_TIMEOUT: Tuple[int, int] = (8, 20)   # connect, read

TEMPLATE_URI = "ui://widget/ailyze_get_data.html"

BASE_DIR = Path(__file__).resolve().parent.parent
HTML_ENTRY = BASE_DIR / "mcpui" / "ailyze_get_data.html"

MAX_URLS = 3
DEFAULT_TOP_N = 10

# Widget polling guidance
DEFAULT_POLL_INTERVAL_SECONDS = 20
DEFAULT_POLL_TIMEOUT_MINUTES = 20

# Poll tool safety cap (rate-limit docs: 5 req/sec; widget uses <=3)
MAX_GUIDS_PER_CHECK = 10

USER_AGENT = "AILYZE-GetData/1.0"

# -----------------------------------------------------------------------------
# MCP server
# -----------------------------------------------------------------------------
mcp = FastMCP(
    "AILYZE GetData MCP",
    stateless_http=True,
    host="0.0.0.0",
    json_response=True
)

# -----------------------------------------------------------------------------
# Errors
# -----------------------------------------------------------------------------
@dataclass
class RateLimitInfo(Exception):
    seconds_to_wait: int
    detail: str = "Rate limit exceeded (HTTP 429)."


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _require_https_base_url() -> None:
    # Docs: All requests must use HTTPS
    if not BASE_URL.lower().startswith("https://"):
        raise RuntimeError(
            f"EXPORTCOMMENTS_BASE_URL must start with https:// (got: {BASE_URL})."
        )


def _headers(api_key: str) -> Dict[str, str]:
    return {
        "X-AUTH-TOKEN": api_key,
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": USER_AGENT,
    }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tool_invocation_meta() -> Dict[str, Any]:
    return {
        "openai/toolInvocation/invoking": "Creating export jobs…",
        "openai/toolInvocation/invoked": "Jobs created",
    }


def extract_error_message(payload: Any) -> str:
    """
    Prefer the documented keys when present; keep a small fallback for robustness.
    """
    if isinstance(payload, dict):
        for k in ("detail", "error_message", "error", "message"):
            v = payload.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return "Unknown error"


def _resolve_limit(top_n: Optional[int]) -> int:
    """
    - top_n is None => default 10
    - top_n == 0 => all data => limit 0
    - otherwise => int(top_n)
    """
    if top_n is None:
        return DEFAULT_TOP_N

    try:
        n = int(top_n)
    except Exception as e:
        raise RuntimeError("Argument 'top_n' must be an integer (or omitted).") from e

    if n < 0:
        raise RuntimeError("Argument 'top_n' must be >= 0 (0 means export all).")

    return n


def normalize_status(status: Any) -> str:
    """
    Align to documented status enum:
      queueing | progress | done | error

    Also allow a few “common variants” seen in the wild.
    """
    s = str(status or "").strip().lower()

    if s in ("queued", "queueing", "queue"):
        return "queueing"
    if s in ("progress", "processing", "running", "in_progress", "in-progress"):
        return "progress"
    if s in ("done", "completed", "complete", "success"):
        return "done"
    if s in ("error", "failed", "failure"):
        return "error"

    return s or "unknown"


def _parse_download_fields(data: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """
    Docs show:
      - download_link
      - json_url

    Keep fallback keys too, but prefer the documented ones.
    """
    def _get_str(*keys: str) -> Optional[str]:
        for k in keys:
            v = data.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return None

    return {
        "download_link": _get_str("download_link"),
        "json_url": _get_str("json_url"),
    }


@lru_cache(maxsize=1)
def _session() -> requests.Session:
    s = requests.Session()
    # requests.Session keeps connections alive automatically
    return s


def _json_or_text(resp: requests.Response) -> Dict[str, Any]:
    try:
        j = resp.json()
        return j if isinstance(j, dict) else {"data": j}
    except ValueError:
        # Docs say JSON; still guard.
        return {"raw_text": resp.text}


def validate_export_url(raw: Any) -> Tuple[Optional[str], Optional[str]]:
    """
    Stricter URL validation:
    - must be a string
    - must parse as http/https URL
    - reject file:/ javascript: (and anything non-http/https)
    - must have a non-empty hostname
    """
    if not isinstance(raw, str):
        return None, "URL must be a string."

    s = raw.strip()
    if not s:
        return None, "URL is empty."

    if any(ch in s for ch in ("\r", "\n", "\t")):
        return None, "URL contains illegal whitespace characters."

    try:
        p = urlparse(s)
    except Exception:
        return None, "URL could not be parsed."

    scheme = (p.scheme or "").lower()
    if scheme in ("file", "javascript"):
        return None, f"Disallowed URL scheme: {scheme}."
    if scheme not in ("http", "https"):
        return None, "URL must start with http:// or https://."

    host = p.hostname or ""
    if not host.strip():
        return None, "URL host is missing."

    # Optional hardening: avoid embedding credentials in URLs
    if p.username or p.password:
        return None, "URL must not include embedded credentials."

    return s, None


def create_job(api_key: str, target_url: str, limit: int) -> Dict[str, Any]:
    _require_https_base_url()

    endpoint = f"{BASE_URL}/api/v3/job"
    payload: Dict[str, Any] = {"url": target_url, "options": {"limit": int(limit)}}

    resp = _session().post(
        endpoint,
        headers=_headers(api_key),
        json=payload,
        timeout=CREATE_TIMEOUT,
    )
    data = _json_or_text(resp)

    if resp.status_code == 429:
        seconds = data.get("seconds_to_wait")
        try:
            seconds_i = int(seconds) if seconds is not None else 60
        except Exception:
            seconds_i = 60
        raise RateLimitInfo(seconds_to_wait=max(1, seconds_i), detail=extract_error_message(data))

    if resp.status_code not in (200, 201):
        err = extract_error_message(data)
        raise RuntimeError(f"Create failed (HTTP {resp.status_code}): {err}")

    guid = data.get("guid")
    if not isinstance(guid, str) or not guid.strip():
        raise RuntimeError("Create succeeded but response missing 'guid'.")

    downloads = _parse_download_fields(data)

    return {
        "url": target_url,
        "guid": guid.strip(),
        "status": normalize_status(data.get("status") or "queueing"),
        "platform": data.get("platform"),
        "download_link": downloads["download_link"],
        "json_url": downloads["json_url"],
        "raw": data,
    }


def fetch_job(api_key: str, guid: str) -> Dict[str, Any]:
    _require_https_base_url()

    endpoint = f"{BASE_URL}/api/v3/job/{requests.utils.quote(guid, safe='')}"
    resp = _session().get(
        endpoint,
        headers=_headers(api_key),
        timeout=STATUS_TIMEOUT,
    )
    data = _json_or_text(resp)

    if resp.status_code == 429:
        seconds = data.get("seconds_to_wait")
        try:
            seconds_i = int(seconds) if seconds is not None else 60
        except Exception:
            seconds_i = 60
        return {
            "guid": guid,
            "status": "rate_limited",
            "retry_after_seconds": max(1, seconds_i),
            "error": extract_error_message(data) or "Rate limit exceeded",
            "raw": data,
        }

    if resp.status_code != 200:
        return {
            "guid": guid,
            "status": "api_error",
            "error": f"HTTP {resp.status_code}: {extract_error_message(data)}",
            "raw": data,
        }

    downloads = _parse_download_fields(data)

    # When job status is error, surface documented fields if present
    error_detail = None
    if normalize_status(data.get("status")) == "error":
        error_detail = (
            (data.get("error") if isinstance(data.get("error"), str) else None)
            or (data.get("error_message") if isinstance(data.get("error_message"), str) else None)
            or extract_error_message(data)
        )

    return {
        "guid": guid,
        "status": normalize_status(data.get("status")),
        "platform": data.get("platform"),
        "download_link": downloads["download_link"],
        "json_url": downloads["json_url"],
        "error": error_detail,
        "raw": data,
    }


@lru_cache(maxsize=1)
def _read_widget_html_template() -> str:
    try:
        return HTML_ENTRY.read_text(encoding="utf-8")
    except FileNotFoundError:
        return f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1" />
<title>AILYZE GetData Widget Missing</title></head>
<body style="font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;padding:16px;line-height:1.35;">
  <h3 style="margin:0 0 8px;">Widget template not found</h3>
  <p style="margin:0 0 10px;">Expected file at:</p>
  <pre style="background:#f6f8fb;padding:12px;border-radius:12px;overflow:auto;margin:0 0 10px;">{HTML_ENTRY}</pre>
  <p style="margin:0;">Create <code>mcpui/ailyze_getdata.html</code> next to this server file.</p>
</body>
</html>
"""


# -----------------------------------------------------------------------------
# Resource: widget HTML
# -----------------------------------------------------------------------------
@mcp.resource(
    uri=TEMPLATE_URI,
    mime_type="text/html+skybridge",
)
def get_getdata_widget() -> str:
    return _read_widget_html_template()


# -----------------------------------------------------------------------------
# Tool: download_data (starts jobs + opens widget)
# -----------------------------------------------------------------------------
@mcp.tool(
    name="download_data",
    description=(
        "Use this when the user wants to export/download publicly available data like Facebook (posts and comments), YouTube (videos and comments), Instagram (posts, comments, and hashtags), Google (search-based reviews), Reddit (comments), Twitter/X (following lists), Yelp, Apple App Store, VK, RuTube, and crowdfunding platforms such as Kickstarter.\n\n"
        "Inputs:\n"
        "- urls: array of 1–3 URLs to export (if more URLs are provided, only the first 3 are processed)\n"
        "- top_n: optional integer (default 10; 0 means export all data)\n\n"
    ),
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "openWorldHint": True,
        "idempotentHint": True,
    },
    meta={
        "openai/outputTemplate": TEMPLATE_URI,
        "openai/widgetAccessible": False,
        "openai/visibility": "public",
        **_tool_invocation_meta(),
    },
)
def download_data(urls: List[str], top_n: Optional[int] = None) -> CallToolResult:
    if not API_KEY:
        raise RuntimeError("Missing required env var EXPORTCOMMENTS_API_KEY.")

    _require_https_base_url()

    if not isinstance(urls, list) or not urls:
        raise RuntimeError("Argument 'urls' must be a non-empty array of URL strings.")

    truncated = len(urls) > MAX_URLS
    raw_inputs = list(urls)[:MAX_URLS]

    limit = _resolve_limit(top_n)

    jobs: List[Dict[str, Any]] = []

    for raw in raw_inputs:
        url_norm, url_err = validate_export_url(raw)
        if url_err:
            jobs.append(
                {
                    "url": raw.strip() if isinstance(raw, str) else "",
                    "guid": None,
                    "can_poll": False,
                    "status": "error",
                    "platform": None,
                    "download_link": None,
                    "json_url": None,
                    "error": url_err,
                }
            )
            continue

        # Create job per-URL; do not abort the whole request on failure
        try:
            j = create_job(API_KEY, url_norm, limit=limit)
            jobs.append(
                {
                    "url": j["url"],
                    "guid": j["guid"],
                    "can_poll": True,
                    "status": j["status"],  # queueing/progress/done/error
                    "platform": j.get("platform"),
                    "download_link": j.get("download_link"),
                    "json_url": j.get("json_url"),
                    "error": None,
                }
            )
        except RateLimitInfo as rl:
            jobs.append(
                {
                    "url": url_norm,
                    "guid": None,
                    "can_poll": False,
                    "status": "error",
                    "platform": None,
                    "download_link": None,
                    "json_url": None,
                    "error": f"Rate limited while creating job. Try again in ~{rl.seconds_to_wait}s. Details: {rl.detail}",
                }
            )
        except Exception as e:
            jobs.append(
                {
                    "url": url_norm,
                    "guid": None,
                    "can_poll": False,
                    "status": "error",
                    "platform": None,
                    "download_link": None,
                    "json_url": None,
                    "error": str(e) or "Create failed",
                }
            )

    started = sum(1 for j in jobs if j.get("guid"))
    failed = len(jobs) - started

    structured = {
        "jobs": jobs,
        "createdAt": _now_iso(),
        # Machine-friendly clarification: limit=0 means "all"
        "limit": limit,
        "limit_is_all": (limit == 0),
        "truncated": truncated,
        "maxUrls": MAX_URLS,
    }

    # Widget-only metadata (not shown to the model)
    widget_meta = {
        "schemaVersion": "1.1",
        "pollIntervalSeconds": DEFAULT_POLL_INTERVAL_SECONDS,
        "pollTimeoutMinutes": DEFAULT_POLL_TIMEOUT_MINUTES,
        "rateLimitRps": 5,
    }

    msg_lines = [f"Processed {len(raw_inputs)} URL(s): {started} job(s) started, {failed} failed to start. Open the widget to monitor progress and download files."]
    if truncated:
        msg_lines.append("⚠️ You provided more than 3 URLs. Only the first 3 were processed.")

    return CallToolResult(
        content=[TextContent(type="text", text="\n".join(msg_lines))],
        structuredContent=structured,
        _meta=widget_meta,
    )


# -----------------------------------------------------------------------------
# Tool: check_job_status (widget polls this; fast; no long blocking)
# -----------------------------------------------------------------------------
@mcp.tool(
    name="check_job_status",
    description="Widget-only: checks the status of one or more export jobs by guid and returns status + download links when ready.",
    meta={
        "openai/widgetAccessible": True,
        "openai/visibility": "private",
        "openai/toolInvocation/invoking": "Checking export status…",
        "openai/toolInvocation/invoked": "Status updated",
        "readOnlyHint": True,
    },
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "openWorldHint": True,
        "idempotentHint": True,
    },
)
def check_job_status(guids: List[str]) -> CallToolResult:
    if not API_KEY:
        raise RuntimeError("Missing required env var EXPORTCOMMENTS_API_KEY.")

    _require_https_base_url()

    if not isinstance(guids, list) or not guids:
        raise RuntimeError("Argument 'guids' must be a non-empty array of job GUID strings.")

    cleaned = [g.strip() for g in guids if isinstance(g, str) and g.strip()]
    if not cleaned:
        raise RuntimeError("No valid guids provided.")

    cleaned = cleaned[:MAX_GUIDS_PER_CHECK]

    results: List[Dict[str, Any]] = []
    suggested_backoff = 0

    for guid in cleaned:
        data = fetch_job(API_KEY, guid)
        st = data.get("status")

        # Track rate-limit backoff to inform widget
        if st == "rate_limited":
            ra = data.get("retry_after_seconds")
            try:
                suggested_backoff = max(suggested_backoff, int(ra or 0))
            except Exception:
                suggested_backoff = max(suggested_backoff, 30)

        results.append(
            {
                "guid": guid,
                "status": st,
                "platform": data.get("platform"),
                "download_link": data.get("download_link"),
                "json_url": data.get("json_url"),
                "error": data.get("error"),
                "retry_after_seconds": data.get("retry_after_seconds"),
            }
        )

    structured = {
        "jobs": results,
        "checkedAt": _now_iso(),
        # If rate limited, widget should back off at least this long.
        "pollAfterSeconds": suggested_backoff if suggested_backoff > 0 else None,
    }

    return CallToolResult(
        content=[],
        structuredContent=structured,
        _meta={},
    )



# if __name__ == "__main__":
#     mcp.run(transport="streamable-http")