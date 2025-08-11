import json
from typing import AsyncIterator, Optional

from jupyter_client import AsyncKernelManager

try:
    import bleach  # type: ignore
except Exception:  # pragma: no cover
    bleach = None  # sanitize becomes a no-op


class KernelPool:
    """Simple per-notebook kernel pool.
    One kernel per notebook path, lazy-started on first execution.
    """

    def __init__(self) -> None:
        self._by_nb: dict[str, AsyncKernelManager] = {}

    async def _get_or_start(self, nb_path: str) -> AsyncKernelManager:
        km = self._by_nb.get(nb_path)
        if km is None:
            km = AsyncKernelManager(kernel_name="python3")
            await km.start_kernel()
            self._by_nb[nb_path] = km
        return km

    async def execute_stream(self, nb_path: str, code: str) -> AsyncIterator[str]:
        """Yield SSE-formatted events while executing `code` in the kernel for `nb_path`.
        Produces `event: output` with pre-rendered HTML chunks, then `event: done`.
        """
        km = await self._get_or_start(nb_path)
        kc = await km.client()
        await kc.start_channels()

        # Start execution
        kc.execute(code, store_history=True)

        # helper to emit a single SSE event with possibly multi-line data
        def sse(event: str, data: str) -> str:
            lines = data.splitlines() or [""]
            return "".join([f"event: {event}\n", *[f"data: {ln}\n" for ln in lines], "\n"])  # end of message

        # Render IOPub -> HTML fragments
        def render_html(msg_content: dict) -> Optional[str]:
            # stream
            if "text" in msg_content and "name" in msg_content:
                name = msg_content.get("name") or "stdout"
                text = msg_content.get("text", "")
                cls = "stderr" if name == "stderr" else "stdout"
                return f'<pre class="stream {cls}">{_html(text)}</pre>'
            # error
            if {"ename", "evalue"}.issubset(msg_content.keys()):
                tb = "\n".join(msg_content.get("traceback", []) or [])
                e = _html(f"{msg_content.get('ename')}: {msg_content.get('evalue')}\n{tb}")
                return f'<pre class="error">{e}</pre>'
            # display/execute_result
            data = msg_content.get("data")
            if isinstance(data, dict):
                if data.get("image/png"):
                    b64 = _join(data["image/png"])  # list or str
                    return f'<img alt="png" src="data:image/png;base64,{b64}" />'
                if data.get("text/html"):
                    html = _join(data["text/html"])  # may be list
                    return _sanitize(html)
                if data.get("text/plain"):
                    txt = _join(data["text/plain"])  # may be list
                    return f"<pre>{_html(txt)}</pre>"
            return None

        # Pull IOPub until we see idle
        while True:
            msg = await kc.get_iopub_msg(timeout=60)
            t = msg["header"]["msg_type"]
            if t == "status" and msg["content"].get("execution_state") == "idle":
                break
            if t in ("stream", "error", "display_data", "execute_result"):
                html = render_html(msg["content"])
                if html:
                    yield sse("output", html)

        yield sse("done", "")


def _join(v: object) -> str:
    if isinstance(v, list):
        return "".join(str(x) for x in v)
    return str(v)


def _html(s: str) -> str:
    # minimal escape (we do stronger sanitize when rendering HTML)
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _sanitize(html: str) -> str:
    if not bleach:
        return html
    return bleach.clean(
        html,
        tags=[
            "div",
            "span",
            "pre",
            "code",
            "img",
            "svg",
            "p",
            "em",
            "strong",
            "table",
            "thead",
            "tbody",
            "tr",
            "td",
            "th",
            "ul",
            "ol",
            "li",
            "br",
            "hr",
            "a",
        ],
        attributes={"img": ["src", "alt", "title", "width", "height"], "a": ["href", "title"], "*": ["class"]},
        protocols=["http", "https", "data"],
        strip=True,
    )


