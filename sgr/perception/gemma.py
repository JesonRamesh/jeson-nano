"""Gemma via Ollama (spec §7.3): the fixed prompt, a JSON-schema answer, one blocking call.

Decisions from the hour-1 bench (sgr/perception/bench/bench_results.md):
- model gemma4:e4b (config.GEMMA_MODEL_TAG)
- send the FULL 320x240 frame: e4b rejects the decoy 5/5 on full frames but only 1/5 on 2x crops
- timeout 8 s wall (measured full-frame p50 5.0 s, p95 6.3 s after warm-up); config.GEMMA_TIMEOUT_S
"""

from __future__ import annotations

import base64
import io
import json
import os
import time

import httpx
import numpy as np
from PIL import Image

from sgr import config
from sgr.contracts import GemmaResult

PROMPT = (
    "You are the vision system of a search-and-rescue drone in a burning building. "
    "Look at the image. Answer only JSON: "
    '{"fire": bool, "person": bool, "confidence": 0..1, "caption": "≤12 words"}.'
)
SCHEMA = {
    "type": "object",
    "properties": {
        "fire": {"type": "boolean"},
        "person": {"type": "boolean"},
        "confidence": {"type": "number"},
        "caption": {"type": "string"},
    },
    "required": ["fire", "person", "confidence", "caption"],
}
TIMEOUT_FLOOR_S = 8.0  # measured after warm-up: e4b full frame p50 5.0 s, p95 6.3 s (demo laptop)


def chat_url(remote: bool = False) -> str:
    """local: config.OLLAMA_URL. remote: $SGR_OLLAMA_URL, e.g. http://192.168.1.20:11434/api/chat."""
    if remote:
        url = os.environ.get("SGR_OLLAMA_URL")
        if not url:
            raise ValueError("remote perception needs SGR_OLLAMA_URL=http://<demo-laptop-ip>:11434/api/chat")
        return url
    return config.OLLAMA_URL


def png_b64(frame: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class GemmaTimeout(Exception):
    pass


class GemmaClient:
    def __init__(self, url: str | None = None, model_tag: str | None = None, timeout_s: float | None = None):
        self.url = url or config.OLLAMA_URL
        self.model_tag = model_tag or config.GEMMA_MODEL_TAG
        self.timeout_s = timeout_s or max(config.GEMMA_TIMEOUT_S, TIMEOUT_FLOOR_S)
        self._http = httpx.Client()

    def warm(self, timeout_s: float = 180.0, image_calls: int = 2) -> bool:
        """Load the model, keep it resident (keep_alive -1) and warm the vision path. Call once at startup.

        Measured on the demo laptop: loading takes ~12 s, but the first image request after it still
        takes ~17 s, and the next ones settle at 4-6 s. So also send a few throwaway image requests.
        """
        base = self.url.rsplit("/api/", 1)[0]
        try:
            self._http.post(f"{base}/api/generate", json={"model": self.model_tag, "keep_alive": -1},
                            timeout=timeout_s).raise_for_status()
            saved, self.timeout_s = self.timeout_s, timeout_s
            try:
                dummy = np.full((240, 320, 3), 60, np.uint8)
                for _ in range(image_calls):
                    self.ask(dummy)
            finally:
                self.timeout_s = saved
            return True
        except Exception:
            return False

    def ask(self, frame: np.ndarray) -> GemmaResult:
        """Blocking. Raises GemmaTimeout after timeout_s, or another exception on failure."""
        body = {
            "model": self.model_tag,
            "stream": False,
            "keep_alive": -1,
            "think": False,  # gemma4 thinks by default: ~645 tokens / ~7.8 s per frame vs ~35 / ~2.4 s without
            "format": SCHEMA,
            "options": {"temperature": 0, "num_ctx": 2048},
            "messages": [{"role": "user", "content": PROMPT, "images": [png_b64(frame)]}],
        }
        t0 = time.perf_counter()
        try:
            # a little slack: LivePerception enforces the real wall-clock deadline itself
            r = self._http.post(self.url, json=body, timeout=httpx.Timeout(self.timeout_s + 2.0, connect=3.0))
        except httpx.TimeoutException as e:
            raise GemmaTimeout(str(e)) from e
        r.raise_for_status()
        ans = json.loads(r.json()["message"]["content"])
        return GemmaResult(
            fire=bool(ans["fire"]), person=bool(ans["person"]),
            confidence=float(min(1.0, max(0.0, ans["confidence"]))), caption=str(ans["caption"])[:120],
            latency_s=round(time.perf_counter() - t0, 3), model_tag=self.model_tag,
        )

    def close(self) -> None:
        self._http.close()
