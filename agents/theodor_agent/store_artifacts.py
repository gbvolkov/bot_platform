
from typing import Annotated, Any, Dict, List, Literal, NotRequired, Optional, TypedDict, Union, Callable
import html
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

from platform_utils.storage_svc import upload_and_get_link


_HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Artifacts</title>
<style>
@page {{ size: A4; margin: 20mm; }}
body {{
  font-family: "Segoe UI", Arial, sans-serif;
  font-size: 12pt;
  line-height: 1.45;
  color: #111;
}}
h1 {{ font-size: 20pt; margin: 0 0 12pt; }}
h2 {{ font-size: 16pt; margin: 14pt 0 8pt; }}
h3 {{ font-size: 13pt; margin: 12pt 0 6pt; }}
p {{ margin: 0 0 10pt; }}
pre {{
  border: 1px solid #d0d0d0;
  background: #f6f6f6;
  padding: 10pt;
  border-radius: 4pt;
  overflow: auto;
}}
code {{
  font-family: "Consolas", "Courier New", monospace;
  font-size: 10.5pt;
}}
blockquote {{
  border-left: 3px solid #ccc;
  margin: 8pt 0;
  padding: 2pt 8pt;
  color: #555;
}}
table {{
  border-collapse: collapse;
  margin: 8pt 0;
  width: 100%;
}}
th, td {{
  border: 1px solid #d0d0d0;
  padding: 4pt 6pt;
  vertical-align: top;
}}
.spoiler {{
  background: #333;
  color: #fff;
  border-radius: 2px;
  padding: 0 2px;
}}
section.chapter {{
  page-break-before: always;
  break-before: page;
}}
section.chapter:first-of-type {{
  page-break-before: auto;
  break-before: auto;
}}
</style>
</head>
<body>
{content}
</body>
</html>
"""


_MARKER_SPECS = {
    "spoiler": {"open": '<span class="spoiler">', "close": "</span>", "literal": "||"},
    "underline": {"open": "<u>", "close": "</u>", "literal": "__"},
    "bold": {"open": "**", "close": "**", "literal": "*"},
    "italic": {"open": "_", "close": "_", "literal": "_"},
    "strike": {"open": "~~", "close": "~~", "literal": "~"},
}


def _escape_literal(text: str) -> str:
    return "".join(f"&#{ord(ch)};" for ch in text)


def _has_closing_marker(text: str, start: int, marker: str) -> bool:
    literal = _MARKER_SPECS[marker]["literal"]
    length = len(literal)
    idx = start
    size = len(text)
    while idx < size:
        ch = text[idx]
        if ch == "\\":
            idx += 2
            continue
        if length == 2 and text.startswith(literal, idx):
            return True
        if length == 1 and ch == literal:
            return True
        idx += 1
    return False


def _toggle_marker(
    *,
    marker: str,
    text: str,
    idx: int,
    out: List[str],
    stack: List[tuple[str, int, str]],
) -> int:
    spec = _MARKER_SPECS[marker]
    literal = spec["literal"]
    length = len(literal)
    if stack and stack[-1][0] == marker:
        out.append(spec["close"])
        stack.pop()
        return idx + length
    if any(item[0] == marker for item in stack):
        out.append(_escape_literal(literal))
        return idx + length
    if not _has_closing_marker(text, idx + length, marker):
        out.append(_escape_literal(literal))
        return idx + length
    out.append(spec["open"])
    stack.append((marker, len(out) - 1, literal))
    return idx + length


def _process_inline(text: str) -> str:
    out: List[str] = []
    stack: List[tuple[str, int, str]] = []
    idx = 0
    size = len(text)
    while idx < size:
        ch = text[idx]
        if ch == "\\" and idx + 1 < size:
            out.append(_escape_literal(text[idx + 1]))
            idx += 2
            continue
        if text.startswith("||", idx):
            idx = _toggle_marker(marker="spoiler", text=text, idx=idx, out=out, stack=stack)
            continue
        if text.startswith("__", idx):
            idx = _toggle_marker(marker="underline", text=text, idx=idx, out=out, stack=stack)
            continue
        if ch == "*":
            idx = _toggle_marker(marker="bold", text=text, idx=idx, out=out, stack=stack)
            continue
        if ch == "_":
            idx = _toggle_marker(marker="italic", text=text, idx=idx, out=out, stack=stack)
            continue
        if ch == "~":
            idx = _toggle_marker(marker="strike", text=text, idx=idx, out=out, stack=stack)
            continue
        out.append(ch)
        idx += 1
    while stack:
        marker, out_idx, literal = stack.pop()
        out[out_idx] = _escape_literal(literal)
    return "".join(out)


def _find_next_unescaped_backtick(text: str, start: int) -> int:
    idx = start
    size = len(text)
    while idx < size:
        if text[idx] == "`" and (idx == 0 or text[idx - 1] != "\\"):
            return idx
        idx += 1
    return size


def _normalize_markdown_v2(text: str) -> str:
    if not text:
        return ""
    out: List[str] = []
    idx = 0
    size = len(text)
    while idx < size:
        if text.startswith("```", idx) and (idx == 0 or text[idx - 1] != "\\"):
            end = text.find("```", idx + 3)
            if end == -1:
                out.append(text[idx:])
                break
            end += 3
            out.append(text[idx:end])
            idx = end
            continue
        if text[idx] == "`" and (idx == 0 or text[idx - 1] != "\\"):
            end = idx + 1
            while end < size:
                if text[end] == "`" and text[end - 1] != "\\":
                    break
                end += 1
            if end < size:
                out.append(text[idx : end + 1])
                idx = end + 1
                continue
        next_tick = _find_next_unescaped_backtick(text, idx)
        chunk = text[idx:next_tick]
        out.append(_process_inline(chunk))
        idx = next_tick
    return "".join(out)


def _markdown_v2_to_html(text: str) -> str:
    try:
        from mistletoe import Document
        from mistletoe.html_renderer import HtmlRenderer
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("mistletoe is required to render MarkdownV2.") from exc

    class _HardBreakRenderer(HtmlRenderer):
        def render_line_break(self, token) -> str:
            return "<br />\n"

    normalized = _normalize_markdown_v2(text)
    with _HardBreakRenderer() as renderer:
        return renderer.render(Document(normalized)).strip()


def _build_html_document(entries: List[Dict[str, str]]) -> str:
    sections: List[str] = []
    for entry in entries:
        title = html.escape(entry["title"])
        body_html = _markdown_v2_to_html(entry["body"])
        if body_html:
            content = body_html
        else:
            content = "<p></p>"
        sections.append(
            f'<section class="chapter"><h1>{title}</h1>\n{content}\n</section>'
        )
    return _HTML_TEMPLATE.format(content="\n".join(sections))


def _find_browser_executable() -> str:
    candidates = [
        shutil.which("msedge"),
        shutil.which("chrome"),
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
    raise FileNotFoundError(
        "No Chromium-based browser found (Edge or Chrome) for PDF rendering."
    )


def _render_pdf(html_document: str, output_path: Path) -> None:
    browser = _find_browser_executable()
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp_dir:
        html_path = Path(tmp_dir) / "artifacts.html"
        html_path.write_text(html_document, encoding="utf-8")
        url = html_path.resolve().as_uri()
        cmd = [
            browser,
            "--headless",
            "--disable-gpu",
            "--no-first-run",
            "--no-default-browser-check",
            "--print-to-pdf-no-header",
            f"--print-to-pdf={output_path}",
            url,
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not output_path.exists():
        raise RuntimeError("PDF rendering failed: output file was not created.")


def _resolve_output_path() -> Path:
    configured = os.getenv("THEODOR_ARTIFACTS_PDF_PATH")
    store_path = Path(configured) if configured else Path.cwd()
    store_path.mkdir(parents=True, exist_ok=True)
    if not os.access(store_path, os.W_OK):
        raise PermissionError(f"Artifacts store path is not writable: {store_path}")
    unique_name = f"artifacts_{uuid.uuid4().hex}.pdf"
    return store_path / unique_name


def store_artifacts(artifacts: Any) -> str:
    if isinstance(artifacts, dict):
        artifact_items = ((k, artifacts[k]) for k in sorted(artifacts))
    else:
        artifact_items = enumerate(artifacts)

    parts: List[str] = []
    entries: List[Dict[str, str]] = []
    append = parts.append
    for artifact_id, details in artifact_items:
        details = details or {}
        definition = details.get("artifact_definition") or {}
        name = definition.get("name") or f"Artifact {artifact_id + 1}"
        append(f"## {artifact_id + 1}. {name}\n")
        body_text = (details.get("artifact_final_text") or "").strip()
        append(body_text)
        append("")
        entries.append(
            {
                "title": f"{artifact_id + 1}. {name}",
                "body": body_text,
            }
        )
    
    final_text = "\n".join(parts).rstrip() + "\n"
    #out_path.write_text(final_text, encoding="utf-8")
    print(final_text)

    if entries:
        output_path = _resolve_output_path()
        html_document = _build_html_document(entries)
        _render_pdf(html_document, output_path)
        user_url = upload_and_get_link(str(output_path))
        print(f"PDF saved to: {user_url}")
        return str(user_url)
