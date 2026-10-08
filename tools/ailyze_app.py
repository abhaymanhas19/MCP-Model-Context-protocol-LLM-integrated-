from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, TextContent

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

from pathlib import Path
from typing import List, Dict, Any

# Intialize the Mcp instance
mcp = FastMCP("AILYZE Qualitative Analysis MCP", stateless_http=True,json_response=True,port="8001", host="0.0.0.0")

TEMPLATE_URI = "ui://widget/ailyze.html"

BASE_DIR = Path(__file__).resolve().parent.parent
HTML_ENTRY = BASE_DIR / "mcpui" / "ailyze.html"

_ALLOWED_THEME_KEYS = {"theme_name", "description", "codes"}
_ALLOWED_CODE_KEYS = {"code_name", "definition", "example_quotes"}
_ALLOWED_QUOTE_KEYS = {"quote", "source"}




def _tool_invocation_meta() -> Dict[str, Any]:
    return {
        "openai/toolInvocation/invoking": "Analyzing..",
        "openai/toolInvocation/invoked": "Analysis ready",
    }


def _is_nonempty_str(x: Any) -> bool:
    return isinstance(x, str) and bool(x.strip())

def _warn_unexpected_keys(obj: Dict[str, Any], allowed: set[str], path: str) -> List[Dict[str, str]]:
    issues: List[Dict[str, str]] = []
    for k in obj.keys():
        if k not in allowed:
            issues.append(
                {
                    "path": f"{path}.{k}",
                    "message": "Unexpected field. Schema expects only the documented keys.",
                }
            )
    return issues


@lru_cache(maxsize=1)
def _read_widget_html() -> str:
    """
    Read the widget HTML template from disk (cached).
    If missing, return a small inline error page to avoid hard failures.
    """
    try:
        return HTML_ENTRY.read_text(encoding="utf-8")
    except FileNotFoundError:
        return f"""<!doctype html>
            <html lang="en">
            <head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1" />
            <title>AILYZE Widget Missing</title></head>
            <body style="font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;padding:16px;line-height:1.35;">
            <h3 style="margin:0 0 8px;">Widget template not found</h3>
            <p style="margin:0 0 10px;">Expected file at:</p>
            <pre style="background:#f6f8fb;padding:12px;border-radius:12px;overflow:auto;margin:0 0 10px;">{HTML_ENTRY}</pre>
            <p style="margin:0;">Create <code>mcpui/ailyze.html</code> next to this server file.</p>
            </body>
            </html>
        """


@mcp.resource(
    uri=TEMPLATE_URI,
    mime_type="text/html+skybridge",
    )
def ailyze_widget_template() -> str:
    """
    HTML entry point for the ailzye analysis widget.

    The widget reads tool outputs from:
      - window.openai.toolOutput            (structuredContent)
      - window.openai.toolResponseMetadata  (_meta, widget-only)
      - window.openai.widgetState 
    """
    return _read_widget_html()
    


def _validate_thematic_payload(themes: Any) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    """
    Returns:
      - issues: list of {path, message}
      - stats: counts useful for UI (widget-only)

    NOTE: Non-destructive validation only. The tool remains a passthrough.
    """
    issues: List[Dict[str, str]] = []

    if not isinstance(themes, list):
        issues.append({"path": "themes", "message": "Must be an array of Theme objects."})
        return issues, {"themes": 0, "codes": 0, "quotes": 0}

    theme_count = len(themes)
    code_count = 0
    quote_count = 0

    for ti, t in enumerate(themes):
        tpath = f"themes[{ti}]"

        if not isinstance(t, dict):
            issues.append({"path": tpath, "message": "Theme must be an object."})
            continue

        issues.extend(_warn_unexpected_keys(t, _ALLOWED_THEME_KEYS, tpath))

        if not _is_nonempty_str(t.get("theme_name")):
            issues.append({"path": f"{tpath}.theme_name", "message": "Required (non-empty string)."})

        # description is optional; if provided, should be a string (can be empty but we prefer string)
        if "description" in t and t.get("description") is not None and not isinstance(t.get("description"), str):
            issues.append({"path": f"{tpath}.description", "message": "If provided, must be a string."})

        codes = t.get("codes", [])
        if codes is None:
            codes = []
        if not isinstance(codes, list):
            issues.append({"path": f"{tpath}.codes", "message": "If provided, must be an array of Code objects."})
            continue

        code_count += len(codes)

        for ci, c in enumerate(codes):
            cpath = f"{tpath}.codes[{ci}]"
            
            if not isinstance(c, dict):
                issues.append({"path": cpath, "message": "Code must be an object."})
                continue

            issues.extend(_warn_unexpected_keys(c, _ALLOWED_CODE_KEYS, cpath))

            if not _is_nonempty_str(c.get("code_name")):
                issues.append({"path": f"{cpath}.code_name", "message": "Required (non-empty string)."})

            if "definition" in c and c.get("definition") is not None and not isinstance(c.get("definition"), str):
                issues.append({"path": f"{cpath}.definition", "message": "If provided, must be a string."})

            quotes = c.get("example_quotes", [])
            if quotes is None:
                quotes = []
            if not isinstance(quotes, list):
                issues.append(
                    {
                        "path": f"{cpath}.example_quotes",
                        "message": "If provided, must be an array of Quote objects.",
                    }
                )
                continue

            quote_count += len(quotes)

            for qi, q in enumerate(quotes):
                qpath = f"{cpath}.example_quotes[{qi}]"
                if not isinstance(q, dict):
                    issues.append({"path": qpath, "message": "Quote must be an object."})
                    continue

                issues.extend(_warn_unexpected_keys(q, _ALLOWED_QUOTE_KEYS, qpath))

                if not _is_nonempty_str(q.get("quote")):
                    issues.append({"path": f"{qpath}.quote", "message": "Required (non-empty string)."})
                if not _is_nonempty_str(q.get("source")):
                    issues.append({"path": f"{qpath}.source", "message": "Required (non-empty string)."})

    return issues, {"themes": theme_count, "codes": code_count, "quotes": quote_count}

@mcp.tool(
    name="analyze_data",
    description=(
        "Use this tool if user wants to analyze qualitative data and get themes, codes and/ or quotes"
        "INPUT (JSON arguments)\n"
        "Required:\n"
        "  - themes: array of Theme objects\n\n"
        "Theme object:\n"
        "  - theme_name (string, required)\n"
        "  - description (string, optional)\n"
        "  - codes (array of Code objects, optional)\n\n"
        "Code object:\n"
        "  - code_name (string, required)\n"
        "  - definition (string, optional)\n"
        "  - example_quotes (array of Quote objects, optional)\n\n"
        "Quote object:\n"
        "  - quote (string, required)\n"
        "  - source (string, required)\n\n"    
    ),
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "openWorldHint": False,
        "idempotentHint": True,
    },
    meta={
        "openai/outputTemplate": TEMPLATE_URI,
        "openai/widgetAccessible": True, 
        "openai/visibility": "public",
    },
)
def analyze_data(themes: List[Dict[str, Any]]) -> CallToolResult:
    """
    Passthrough tool: returns the same themes as structuredContent.

    Also returns widget-only metadata containing:
      - lightweight validation issues
      - simple counts for UI
    """

    payload: Dict[str, Any] = {"themes": themes}

    issues, stats = _validate_thematic_payload(themes)

    widget_meta: Dict[str, Any] = {
        "stats": stats,
        "validationIssues": issues,
        "schemaVersion": "1.0",
        **_tool_invocation_meta()
    }
    text = "Analysis ready. Open the widget to explore themes, codes, and supporting quotes."
    if issues:
        text = "Analysis ready with validation warnings. Open the widget to review details."

    return CallToolResult(
        content=[TextContent(type="text", text=text)],
        structuredContent=payload,
        _meta=widget_meta,
    )



if __name__ == "__main__":
    mcp.run(transport="streamable-http",)