"""Exercise the shipped web client in an installed browser, without agent/model calls."""
from __future__ import annotations

import base64
import os
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pytest

playwright = pytest.importorskip("playwright.sync_api")
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def browser() -> Iterator[Any]:
    with playwright.sync_playwright() as runtime:
        candidates = [Path(runtime.chromium.executable_path),
            Path(os.environ.get("PROGRAMFILES", "C:/Program Files")) / "Google/Chrome/Application/chrome.exe",
            Path(os.environ.get("PROGRAMFILES(X86)", "C:/Program Files (x86)")) / "Microsoft/Edge/Application/msedge.exe"]
        executable = next((path for path in candidates if path.is_file()), None)
        if executable is None:
            pytest.skip("No installed Chromium browser; no browser dependency is downloaded by these tests")
        instance = runtime.chromium.launch(executable_path=str(executable), headless=True)
        yield instance
        instance.close()


@pytest.fixture
def chat(browser: Any) -> Iterator[tuple[Any, list[dict[str, Any]], dict[str, Any]]]:
    context = browser.new_context(accept_downloads=True)
    replies: list[dict[str, Any]] = []
    control: dict[str, Any] = {"defer": set(), "held": [], "create_replies": [], "create_count": 0}
    page = context.new_page()
    page.add_init_script("window.fixtureHeldRequests = []")

    def route_request(route: Any) -> None:
        parsed = urlparse(route.request.url)
        if parsed.hostname == "127.0.0.1":
            route.continue_()
            return
        path = parsed.path
        if path in control["defer"]:
            control["held"].append((path, route))
            page.evaluate("(path) => window.fixtureHeldRequests.push(path)", path)
            return
        if path == "/":
            route.fulfill(content_type="text/html", body=(ROOT / "web_chat/templates/index.html").read_text(encoding="utf-8"))
        elif path == "/static/chat.js":
            route.fulfill(content_type="application/javascript", body=(ROOT / "web_chat/static/chat.js").read_text(encoding="utf-8"))
        elif path == "/static/styles.css":
            route.fulfill(content_type="text/css", body=(ROOT / "web_chat/static/styles.css").read_text(encoding="utf-8"))
        elif path == "/api/agents":
            route.fulfill(json=[{"id": "fixture", "name": "Offline fixture"}])
        elif path == "/api/conversations":
            control["create_count"] += 1
            response = control["create_replies"].pop(0) if control["create_replies"] else {
                "id": f"fixture-conversation-{control['create_count']}", "status": "active"}
            route.fulfill(status=201 if response["status"] == "active" else 202, json=response)
        elif path.startswith("/api/conversations/") and path.endswith("/messages"):
            assert replies, "Test must provide a response before submitting a message"
            route.fulfill(json={"agent_message": replies.pop(0)})
        elif path.startswith("/api/conversations/"):
            route.fulfill(json={"id": path.rsplit("/", 1)[-1], "status": "active"})
        else:
            route.abort()

    page.route("**/*", route_request)
    page.goto("http://chat.test/")
    page.wait_for_function("document.querySelector('#agent-select').options.length === 1")
    yield page, replies, control
    context.close()


@pytest.fixture
def supplied_file_url() -> Iterator[str]:
    class FileHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            body = b"Existing supplied URL\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Disposition", 'attachment; filename="supplied.txt"')
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), FileHandler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    yield f"http://127.0.0.1:{server.server_port}/supplied.txt"
    server.shutdown()
    server.server_close()
    worker.join()


def payload(filename: str, data: bytes, content_type: str = "text/markdown") -> dict[str, str]:
    return {"type": "file", "filename": filename, "content_type": content_type,
            "data": base64.b64encode(data).decode("ascii")}


def submit(page: Any, replies: list[dict[str, Any]], message: dict[str, Any]) -> None:
    replies.append(message)
    previous = page.locator(".message.assistant").count()
    page.locator("#message-input").fill("Offline attachment contract check")
    page.locator("#message-form button").click()
    page.wait_for_function("!state.isSending")
    assert not replies
    assert page.locator(".message.assistant").count() == previous + 1


def observe_finished_requests(page: Any) -> None:
    page.evaluate("""() => {
        window.finishedRequestGenerations = [];
        const originalSend = sendMessage;
        sendMessage = (...args) => {
            const request = originalSend(...args);
            const settled = () => window.finishedRequestGenerations.push(args[1]);
            request.then(settled, settled);
            return request;
        };
    }""")


def test_multiple_real_downloads_and_verbatim_prose(chat: Any, tmp_path: Path) -> None:
    page, replies, _ = chat
    body = "Full result. [invented.md](attachment)\n<em>Leave this text untouched.</em>"
    expected = ["# Карта\nРешение: сохранить неопределённость.\n".encode(), b"%PDF-1.4\n\x00\xff\x80binary fixture\n"]
    names = ["карта [пилот].md", "report.pdf"]
    submit(page, replies, {"raw_text": body, "metadata": {"attachments": [
        payload(names[0], expected[0]), payload(names[1], expected[1], "application/pdf")]}})
    assert page.locator(".message.assistant .message-text").text_content() == body
    assert page.locator(".message-text a, .message-text em").count() == 0
    controls = page.locator(".message-attachments a")
    assert controls.count() == 2
    for index in range(2):
        assert controls.nth(index).text_content() == names[index]
        assert controls.nth(index).get_attribute("download") == names[index]
        assert controls.nth(index).get_attribute("href").startswith("blob:http://chat.test/")
        with page.expect_download() as download_info:
            controls.nth(index).click()
        download = download_info.value
        download.save_as(tmp_path / f"actual-{index}.bin")
        assert (tmp_path / f"actual-{index}.bin").read_bytes() == expected[index]


def test_plain_typed_text_overrides_legacy_fallback_and_only_real_files_create_controls(chat: Any, tmp_path: Path) -> None:
    page, replies, _ = chat
    text = "[Invented](https://invented.test/file) <a href='file:///private'>open</a>"
    data = b"Actual code-owned bytes"
    submit(page, replies, {"raw_text": "```\n" + text + "\n```", "content": {"type": "segments", "parts": [
        {"type": "text", "format": "plain", "text": text}]}, "metadata": {"attachments": [payload("actual.md", data)]}})
    assert page.locator(".message-text").last.text_content() == text
    assert page.locator(".message-text a").count() == 0
    control = page.locator(".message-attachments a")
    assert control.count() == 1 and control.get_attribute("href").startswith("blob:")
    with page.expect_download() as download_info:
        control.click()
    download_info.value.save_as(tmp_path / "actual-download.md")
    assert (tmp_path / "actual-download.md").read_bytes() == data


def test_file_only_reply_and_literal_filename(chat: Any) -> None:
    page, replies, _ = chat
    name = "<img src=x onerror=alert(1)>.md"
    submit(page, replies, {"raw_text": "", "metadata": {"attachments": [payload(name, b"actual") ]}})
    assert page.locator(".message.assistant .message-text").text_content() == ""
    assert page.locator(".message-attachments a").text_content() == name
    assert page.locator(".message-attachments img").count() == 0


def test_invalid_items_do_not_hide_real_download(chat: Any, tmp_path: Path) -> None:
    page, replies, _ = chat
    submit(page, replies, {"raw_text": "Files", "metadata": {"attachments": [None,
        {"filename": "broken.bin", "data": "%%%", "url": "http://chat.test/valid-looking-supplied-url"},
        {"filename": "unsafe.bin", "url": "javascript:alert(1)"},
        {"filename": "id-only.bin", "artifact_id": "opaque-id"},
        payload("real.md", b"real file\n")]}})
    assert page.locator(".attachment-error").count() == 4
    assert all("Download unavailable" in text for text in page.locator(".attachment-error").all_text_contents())
    assert page.locator(".message-attachments a").count() == 1
    with page.expect_download() as download_info:
        page.locator(".message-attachments a").click()
    download_info.value.save_as(tmp_path / "actual.md")
    assert (tmp_path / "actual.md").read_bytes() == b"real file\n"


def test_metadata_precedence_and_content_fallback(chat: Any) -> None:
    page, replies, _ = chat
    fallback = {"type": "segments", "parts": [{"type": "text", "text": "not a file"}, payload("fallback.md", b"fallback")]}
    submit(page, replies, {"raw_text": "Metadata owns this set", "metadata": {
        "attachments": [payload("metadata.md", b"metadata")]}, "content": fallback})
    assert page.locator(".message.assistant").last.locator("a").all_text_contents() == ["metadata.md"]
    submit(page, replies, {"raw_text": "An explicit empty set", "metadata": {"attachments": []}, "content": fallback})
    assert page.locator(".message.assistant").last.locator("a").count() == 0
    submit(page, replies, {"raw_text": "Only content parts supplied", "content": fallback})
    assert page.locator(".message.assistant").last.locator("a").all_text_contents() == ["fallback.md"]


def test_real_supplied_url_and_blob_cleanup_on_reset(chat: Any, tmp_path: Path, supplied_file_url: str) -> None:
    page, replies, _ = chat
    page.evaluate("""() => {
        window.revokedAttachmentUrls = [];
        const nativeRevoke = URL.revokeObjectURL.bind(URL);
        URL.revokeObjectURL = (url) => { window.revokedAttachmentUrls.push(url); nativeRevoke(url); };
    }""")
    submit(page, replies, {"raw_text": "Actual files", "metadata": {"attachments": [
        payload("first.md", b"first"), payload("second.md", b"second"),
        {"filename": "supplied.txt", "data": None, "url": supplied_file_url}]}})
    controls = page.locator(".message-attachments a")
    object_urls = [controls.nth(index).get_attribute("href") for index in range(2)]
    assert controls.nth(2).get_attribute("href") == supplied_file_url
    with page.expect_download() as download_info:
        controls.nth(2).click()
    download_info.value.save_as(tmp_path / "supplied.txt")
    assert (tmp_path / "supplied.txt").read_bytes() == b"Existing supplied URL\n"
    page.locator("#new-chat-btn").click()
    assert page.locator(".message").count() == 0
    assert page.evaluate("window.revokedAttachmentUrls") == object_urls
    assert page.evaluate("state.attachmentUrls.size") == 0
    assert page.evaluate("""async (urls) => Promise.all(urls.map(async (url) => {
        try { await fetch(url); return false; } catch { return true; }
    }))""", object_urls) == [True, True]


def test_ordinary_direct_answer_has_no_generated_navigation(chat: Any) -> None:
    page, replies, _ = chat
    text = "An ordinary answer with https://example.com and [literal](attachment)."
    submit(page, replies, {"raw_text": text, "metadata": {}})
    assert page.locator(".message.assistant .message-text").text_content() == text
    assert page.locator(".message.assistant a").count() == 0


def test_malformed_collection_is_visible_instead_of_using_other_parts(chat: Any) -> None:
    page, replies, _ = chat
    submit(page, replies, {"raw_text": "Files could not be presented", "metadata": {
        "attachments": "invalid collection"}, "content": {"type": "segments", "parts": [
            payload("unexpected.md", b"not the selected attachment set")]}})
    assert page.locator(".attachment-error").count() == 1
    assert page.locator(".message.assistant a").count() == 0


@pytest.mark.parametrize("pending_stage", ["create", "readiness", "reply"])
def test_reset_ignores_previous_conversation_response(chat: Any, pending_stage: str) -> None:
    page, replies, control = chat
    observe_finished_requests(page)
    if pending_stage == "create":
        held_path = "/api/conversations"
        held_response = {"id": "old-late-conversation", "status": "active"}
    elif pending_stage == "readiness":
        control["create_replies"].append({"id": "old-pending-conversation", "status": "pending"})
        held_path = "/api/conversations/old-pending-conversation"
        held_response = {"id": "old-pending-conversation", "status": "active"}
    else:
        held_path = "/api/conversations/fixture-conversation-1/messages"
        held_response = {"agent_message": {"raw_text": "Old reply must not appear", "metadata": {
            "attachments": [payload("old.md", b"old file")]}}}
    control["defer"].add(held_path)
    page.locator("#message-input").fill("Previous conversation request")
    with page.expect_request(f"http://chat.test{held_path}"):
        page.locator("#message-form button").click()
    page.wait_for_function("state.isSending")
    assert len(control["held"]) == 1
    page.locator("#new-chat-btn").click()
    control["defer"].clear()
    submit(page, replies, {"raw_text": "Current conversation reply", "metadata": {
        "attachments": [payload("current.md", b"current file")]}})
    current_id = page.evaluate("state.conversationId")
    held_route = control["held"][0][1]
    held_route.fulfill(status=201 if pending_stage == "create" else 200, json=held_response)
    page.wait_for_function("window.finishedRequestGenerations.includes(0)")
    assert page.evaluate("state.conversationId") == current_id
    assert page.evaluate("state.agentReady") is True
    assert page.evaluate("state.isSending") is False
    assert page.locator(".message.assistant").count() == 1
    assert page.locator(".message-attachments a").all_text_contents() == ["current.md"]
    assert page.evaluate("state.attachmentUrls.size") == 1
    assert page.locator(".status").count() == 0


def test_old_response_does_not_clear_new_pending_request(chat: Any) -> None:
    page, _, control = chat
    observe_finished_requests(page)
    for generation in (0, 1):
        path = f"/api/conversations/fixture-conversation-{generation + 1}/messages"
        control["defer"].add(path)
        page.locator("#message-input").fill(f"Request in generation {generation}")
        with page.expect_request(f"http://chat.test{path}"):
            page.locator("#message-form button").click()
        if generation == 0:
            page.locator("#new-chat-btn").click()
    page.wait_for_function("window.fixtureHeldRequests.length === 2")
    assert len(control["held"]) == 2
    control["held"][0][1].fulfill(json={"agent_message": {"raw_text": "Old response", "metadata": {
        "attachments": [payload("old.md", b"old file")]}}})
    page.wait_for_function("window.finishedRequestGenerations.includes(0)")
    assert page.evaluate("state.isSending") is True
    assert page.locator(".status").text_content() == "Assistant is typing…"
    assert page.locator(".message.assistant").count() == 0
    assert page.evaluate("state.attachmentUrls.size") == 0
    control["held"][1][1].fulfill(json={"agent_message": {"raw_text": "Current response", "metadata": {
        "attachments": [payload("current.md", b"current file")]}}})
    page.wait_for_function("window.finishedRequestGenerations.includes(1) && !state.isSending")
    assert page.locator(".message.assistant .message-text").text_content() == "Current response"
    assert page.locator(".message-attachments a").all_text_contents() == ["current.md"]
