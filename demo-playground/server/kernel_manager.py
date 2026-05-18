"""Wraps jupyter_client to manage kernel sessions."""

from __future__ import annotations

import asyncio
import base64
import json
import uuid
from typing import Any

import jupyter_client


class KernelSession:
    def __init__(self) -> None:
        self.id = str(uuid.uuid4())
        self.km = jupyter_client.KernelManager()
        self.km.start_kernel()
        self.kc = self.km.client()
        self.kc.start_channels()
        self.kc.wait_for_ready(timeout=30)

    def shutdown(self) -> None:
        try:
            self.kc.stop_channels()
            self.km.shutdown_kernel(now=True)
        except Exception:
            pass

    async def execute(self, code: str) -> list[dict[str, Any]]:
        """Execute code synchronously and return all outputs."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._execute_sync, code)

    def _execute_sync(self, code: str) -> list[dict[str, Any]]:
        outputs: list[dict[str, Any]] = []
        msg_id = self.kc.execute(code)

        while True:
            try:
                msg = self.kc.get_iopub_msg(timeout=60)
            except Exception:
                break

            msg_type = msg["msg_type"]
            content = msg["content"]

            if msg_type == "stream":
                outputs.append(
                    {"type": "text", "stream": content.get("name", "stdout"), "text": content["text"]}
                )

            elif msg_type in ("display_data", "execute_result"):
                data: dict[str, str] = content.get("data", {})
                out = _classify_rich(data)
                if out:
                    outputs.append(out)

            elif msg_type == "error":
                outputs.append(
                    {
                        "type": "error",
                        "ename": content.get("ename", "Error"),
                        "evalue": content.get("evalue", ""),
                        "traceback": content.get("traceback", []),
                    }
                )

            elif msg_type == "status":
                if content.get("execution_state") == "idle":
                    break

        return outputs

    async def execute_streaming(self, code: str):
        """Async generator that yields output dicts as they arrive."""
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue[dict | None] = asyncio.Queue()

        def _run():
            msg_id = self.kc.execute(code)
            exec_count = 0
            while True:
                try:
                    msg = self.kc.get_iopub_msg(timeout=60)
                except Exception:
                    break

                msg_type = msg["msg_type"]
                content = msg["content"]

                if msg_type == "stream":
                    loop.call_soon_threadsafe(
                        queue.put_nowait,
                        {"type": "stream", "name": content.get("name", "stdout"), "text": content["text"]},
                    )

                elif msg_type in ("display_data", "execute_result"):
                    data = content.get("data", {})
                    out = _classify_rich(data)
                    if out:
                        if msg_type == "execute_result":
                            exec_count = content.get("execution_count", 0)
                        loop.call_soon_threadsafe(queue.put_nowait, out)

                elif msg_type == "error":
                    loop.call_soon_threadsafe(
                        queue.put_nowait,
                        {
                            "type": "error",
                            "ename": content.get("ename", "Error"),
                            "evalue": content.get("evalue", ""),
                            "traceback": content.get("traceback", []),
                        },
                    )

                elif msg_type == "status":
                    if content.get("execution_state") == "idle":
                        loop.call_soon_threadsafe(
                            queue.put_nowait,
                            {"type": "done", "execution_count": exec_count},
                        )
                        break

            loop.call_soon_threadsafe(queue.put_nowait, None)

        loop.run_in_executor(None, _run)

        while True:
            item = await queue.get()
            if item is None:
                break
            yield item


def _classify_rich(data: dict[str, str]) -> dict | None:
    """Pick the best MIME type from a rich display data dict."""
    if "text/latex" in data:
        return {"type": "latex", "latex": data["text/latex"]}
    if "image/png" in data:
        return {"type": "image", "format": "png", "data": data["image/png"]}
    if "image/svg+xml" in data:
        return {"type": "image", "format": "svg", "data": data["image/svg+xml"]}
    if "text/html" in data:
        return {"type": "html", "html": data["text/html"]}
    if "application/json" in data:
        raw = data["application/json"]
        return {"type": "json", "data": json.loads(raw) if isinstance(raw, str) else raw}
    if "text/plain" in data:
        return {"type": "text", "stream": "stdout", "text": data["text/plain"]}
    return None
