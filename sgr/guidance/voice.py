"""Act 2 drone voices (P): the drones talk to the firefighter ("Careful, fire on your right.").

Guidance decides WHEN a drone speaks and WHAT the facts are (`Callout`), from the swarm's beliefs only.
This module turns a callout into a `CalloutMsg`:

- template mode (no API key, `--voice template`, tests): the fixed line, spoken by the browser's local voice.
- Gemini mode: a worker thread asks Gemini to reword the line in that drone's personality (about 0.5 s),
  checks the reworded line keeps the facts (no invented directions, numbers, drones or hazards; otherwise the
  fixed line is used), then voices it with Gemini TTS, a different voice per drone (about 2 s).
  Every step has a deadline. On any error or timeout the drone still speaks: the fixed line, or the reworded
  line without audio (the browser then uses a local voice). The sim loop never waits for the network.

Danger lines ("Stop. Fire ahead.") are never callouts: GuidanceMsg.say stays instant and offline.
The API key comes from the environment (GOOGLE_API_KEY / GEMINI_API_KEY / google_api) or `.env`. It is
never logged.
"""

from __future__ import annotations

import base64
import json
import os
import queue
import re
import struct
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sgr.contracts import CalloutMsg, Severity

TEXT_MODEL = os.environ.get("SGR_VOICE_TEXT_MODEL", "gemini-3.5-flash-lite")  # ~0.5 s; gemini-3.8-flash took 2-6 s
TTS_MODEL = os.environ.get("SGR_VOICE_TTS_MODEL", "gemini-3.8-flash-lite-tts")  # ~1.7-2.3 s per line
API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
TEXT_TIMEOUT_S = 2.0
TTS_TIMEOUT_S = 4.0
STALE_S = 3.0  # waited this long in the queue (Gemini slow / many lines): skip Gemini, say the fixed line now
# only these get reworded by Gemini (+~0.5 s); urgent lines (fire, survivor, handover) go straight to the voice
REWORD_KINDS = {"intro", "progress", "lead_wait", "lead_return"}
QUEUE_MAX = 8

# personality + Gemini TTS prebuilt voice per drone (distinct voices so the firefighter can tell them apart)
PERSONAS: dict[str, tuple[str, str]] = {
    "D1": ("a calm, experienced team leader: steady and reassuring", "Charon"),
    "D2": ("warm and encouraging", "Kore"),
    "D3": ("upbeat and quick", "Puck"),
    "D4": ("precise and softly spoken", "Aoede"),
    "HUD": ("calm incident commander", "Orus"),  # the HUD guidance lines in online mode
}
CACHE_MAX = 300  # voiced lines kept (fixed lines are pre-generated at startup: instant later)


@dataclass
class Callout:
    """What a drone should say. `text` is the fixed line (the fallback); `facts` are all Gemini may use."""

    speaker: str
    kind: str
    text: str
    facts: dict[str, Any] = field(default_factory=dict)
    severity: Severity = "info"
    created: float = field(default_factory=time.monotonic)


def template_msg(c: Callout) -> CalloutMsg:
    return CalloutMsg(speaker=c.speaker, kind=c.kind, text=c.text, severity=c.severity, source="template")


# ---------------------------------------------------------------------- fact check
_DIRECTIONS = {"left", "right", "ahead", "behind", "back", "backwards", "forward", "forwards", "straight",
               "north", "south", "east", "west", "upstairs", "downstairs", "around", "round"}  # not up/down: "heads up"
# hazard / landmark words, grouped: one word of a group in the fixed line allows the whole group
_TOPICS = [
    {"fire", "fires", "flame", "flames", "blaze", "burning", "heat", "hot"},
    {"survivor", "survivors", "victim", "casualty", "person", "patient"},
    {"debris", "rubble", "collapse", "collapsed", "blocked", "blockage"},
    {"gas", "explosion", "explosive"},
    {"exit", "exits", "stair", "stairs", "staircase", "door", "doors", "doorway", "window", "windows",
     "wall", "walls", "corridor", "hallway", "room", "corner", "junction"},
]
_ONES = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
         "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
         "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
         "ninety": 90, "hundred": 100}


def _words(s: str) -> list[str]:
    return re.findall(r"[a-z]+", s.lower())


def _numbers(s: str) -> set[int]:
    """Every number in a line, digits or words ("thirty-two" -> 32)."""
    out = {int(n) for n in re.findall(r"\d+", s)}
    w = _words(s)
    i = 0
    while i < len(w):
        if w[i] in _TENS:
            v = _TENS[w[i]]
            if i + 1 < len(w) and w[i + 1] in _ONES and 0 < _ONES[w[i + 1]] < 10:
                v += _ONES[w[i + 1]]
                i += 1
            out.add(v)
        elif w[i] in _ONES:
            out.add(_ONES[w[i]])
        i += 1
    return out


def _drones(s: str) -> set[str]:
    return {f"D{n}" for n in re.findall(r"\b[Dd]\s?-?([1-9])\b", s)}


def check_line(line: str, c: Callout) -> str | None:
    """The reworded line if it says nothing the fixed line and facts don't; None -> use the fixed line."""
    line = " ".join(line.replace("*", "").replace('"', "").replace("“", "").replace("”", "").split()).strip()
    n = len(line.split())
    if not 3 <= n <= 14:  # short: a long line delays the next one
        return None
    ref = f"{c.text} {json.dumps(c.facts)} {c.speaker}"
    if not set(_words(line)) & _DIRECTIONS <= set(_words(ref)) & _DIRECTIONS:
        return None  # an invented direction ("turn left") could walk the firefighter into danger
    allowed_n = _numbers(ref) | {1}  # "one more corridor" is fine; any other number must be a given fact
    if not _numbers(line) <= allowed_n:
        return None
    if not _drones(line) <= _drones(ref) | {c.speaker}:
        return None
    lw, rw = set(_words(line)), set(_words(ref))
    for topic in _TOPICS:
        if lw & topic and not rw & topic:
            return None  # no invented hazards or landmarks
    return line


# ---------------------------------------------------------------------- API key
def load_api_key(env_file: str | os.PathLike | None = None) -> str | None:
    for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "google_api"):
        if os.environ.get(name):
            return os.environ[name].strip()
    candidates = [Path(env_file)] if env_file else [Path(".env"), Path(__file__).resolve().parents[2] / ".env"]
    for p in candidates:
        if not p.is_file():
            continue
        for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            m = re.match(r"\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)", line)
            if m and m.group(1).lower() in ("google_api", "google_api_key", "gemini_api_key"):
                v = m.group(2).strip().strip('"').strip("'")
                if v:
                    return v
    return None


def _pcm_to_wav(pcm: bytes, rate: int = 24000) -> bytes:
    """Gemini TTS may return raw 16-bit mono PCM (audio/L16); browsers need a WAV header."""
    return struct.pack("<4sI4s4sIHHIIHH4sI", b"RIFF", 36 + len(pcm), b"WAVE", b"fmt ", 16, 1, 1, rate,
                       rate * 2, 2, 16, b"data", len(pcm)) + pcm


# ---------------------------------------------------------------------- the agent
class VoiceAgent:
    """Gemini wording + Gemini TTS on one background thread. submit() never blocks; poll() returns finished lines."""

    def __init__(self, api_key: str, text_model: str = TEXT_MODEL, tts_model: str = TTS_MODEL,
                 tts: bool = True, client: Any | None = None, log: bool = True) -> None:
        import httpx

        self.text_model, self.tts_model, self.tts = text_model, tts_model, tts
        self._client = client or httpx.Client(headers={"x-goog-api-key": api_key})
        self._in: queue.Queue[Callout | None] = queue.Queue(maxsize=QUEUE_MAX)
        self._out: queue.Queue[CalloutMsg] = queue.Queue()
        self._thinking: dict[str, Any] | None = {"thinkingLevel": "minimal"}
        self._epoch = 0  # clear() bumps it: lines queued before a snapshot load are dropped
        self._log = log
        self.stats = {"gemini": 0, "template": 0, "audio": 0, "rejected": 0, "errors": 0, "cached": 0}
        self._cache: dict[tuple[str, str], tuple[str, str]] = {}  # (speaker, text) -> (audio b64, mime)
        self._cache_lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name="sgr-voice", daemon=True)
        self._thread.start()

    @classmethod
    def from_env(cls, **kw: Any) -> VoiceAgent | None:
        key = load_api_key()
        return cls(key, **kw) if key else None

    # ------------------------------------------------------------ public
    def prewarm(self, lines: list[tuple[str, str]]) -> None:
        """Voice fixed lines in the background (HUD instructions, handovers...) so they play instantly later."""
        def run() -> None:
            for speaker, text in lines:
                if self._cached(speaker, text) is None:
                    self._speak(text, speaker)
            if self._log:
                print(f"sgr voice: {len(self._cache)} fixed lines voiced and cached")

        threading.Thread(target=run, name="sgr-voice-prewarm", daemon=True).start()

    def _cached(self, speaker: str, text: str) -> tuple[str, str] | None:
        with self._cache_lock:
            return self._cache.get((speaker, text))

    def submit(self, c: Callout) -> None:
        if c.kind not in REWORD_KINDS:  # said word for word: an already voiced line plays at once
            hit = self._cached(c.speaker, c.text)
            if hit is not None:
                self.stats["cached"] += 1
                self._emit(CalloutMsg(speaker=c.speaker, kind=c.kind, text=c.text, severity=c.severity,
                                      source="template", audio_b64=hit[0], audio_mime=hit[1]))
                return
            if c.speaker == "HUD" and c.severity != "info":  # a warning can't wait for TTS: local voice now
                self._emit(template_msg(c))
                self.prewarm([(c.speaker, c.text)])  # voiced next time
                return
        try:
            self._in.put_nowait(c)
        except queue.Full:  # Gemini is far behind: say it now, plainly
            self._emit(template_msg(c))

    def poll(self) -> list[CalloutMsg]:
        out = []
        while True:
            try:
                out.append(self._out.get_nowait())
            except queue.Empty:
                return out

    def clear(self) -> None:
        self._epoch += 1
        while True:
            try:
                self._in.get_nowait()
            except queue.Empty:
                break
        self.poll()

    def close(self) -> None:
        self._in.put(None)
        self._thread.join(timeout=1.0)

    # ------------------------------------------------------------ worker
    def _emit(self, m: CalloutMsg) -> None:
        self.stats[m.source] += 1
        if m.audio_b64:
            self.stats["audio"] += 1
        self._out.put(m)

    def _run(self) -> None:
        while True:
            c = self._in.get()
            if c is None:
                return
            epoch = self._epoch
            try:
                m = self._voice(c)
            except Exception as e:  # never let the voice thread die
                self.stats["errors"] += 1
                if self._log:
                    print(f"sgr voice: {c.speaker} {c.kind} error {type(e).__name__}; fixed line")
                m = template_msg(c)
            if epoch == self._epoch:
                self._emit(m)

    def _voice(self, c: Callout) -> CalloutMsg:
        if time.monotonic() - c.created > STALE_S:
            return template_msg(c)
        t0 = time.perf_counter()
        text, source = c.text, "template"
        line = self._reword(c) if c.kind in REWORD_KINDS else None
        if line is not None:
            ok = check_line(line, c)
            if ok:
                text, source = ok, "gemini"
            else:
                self.stats["rejected"] += 1
        t1 = time.perf_counter()
        audio, mime = self._speak(text, c.speaker) if self.tts else (None, None)
        t2 = time.perf_counter()
        if self._log:
            print(f"sgr voice: {c.speaker} {c.kind:14s} {source:8s} text {t1 - t0:.2f}s  tts {t2 - t1:.2f}s "
                  f"{'audio' if audio else 'local voice'}: {text!r}")
        return CalloutMsg(speaker=c.speaker, kind=c.kind, text=text, severity=c.severity, source=source,
                          audio_b64=audio, audio_mime=mime)

    def _post(self, model: str, body: dict, timeout: float) -> dict | None:
        try:
            r = self._client.post(API.format(model=model), json=body, timeout=timeout)
        except Exception:
            self.stats["errors"] += 1
            return None
        if r.status_code == 400 and self._thinking and "thinking" in r.text.lower():
            return {"_retry_without_thinking": True}
        if r.status_code != 200:
            self.stats["errors"] += 1
            return None
        return r.json()

    def _reword(self, c: Callout) -> str | None:
        persona = PERSONAS.get(c.speaker, ("calm and clear", ""))[0]
        system = (
            f"You are {c.speaker}, one of four small rescue drones guiding a firefighter through a dark, "
            f"smoke-filled building to a trapped survivor. Personality: {persona}. You talk to the firefighter "
            "over their headset. Reply with ONE short spoken line (4 to 10 words), plain text, no quotes, no emojis. "
            "Say the same thing as the default line, in your own words. Use ONLY the facts given: never add "
            "directions, hazards, places, names or numbers that are not in the facts or the default line."
        )
        user = f"Event: {c.kind}\nFacts: {json.dumps(c.facts)}\nDefault line: {c.text}"
        for _ in range(2):
            gen: dict[str, Any] = {"temperature": 0.9, "maxOutputTokens": 800}
            if self._thinking:
                gen["thinkingConfig"] = self._thinking
            body = {"systemInstruction": {"parts": [{"text": system}]},
                    "contents": [{"role": "user", "parts": [{"text": user}]}], "generationConfig": gen}
            j = self._post(self.text_model, body, TEXT_TIMEOUT_S)
            if j and j.get("_retry_without_thinking"):
                self._thinking = None  # this model has no "minimal" thinking level
                continue
            if not j:
                return None
            try:
                parts = j["candidates"][0]["content"].get("parts", [])
                return "".join(p.get("text", "") for p in parts if not p.get("thought")).strip() or None
            except (KeyError, IndexError):
                return None
        return None

    def _speak(self, text: str, speaker: str) -> tuple[str | None, str | None]:
        hit = self._cached(speaker, text)
        if hit is not None:
            return hit
        voice = PERSONAS.get(speaker, ("", "Kore"))[1] or "Kore"
        body = {"contents": [{"parts": [{"text": text}]}],  # just the line: style prompts get read out loud
                "generationConfig": {"responseModalities": ["AUDIO"],
                                     "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}}}}
        j = self._post(self.tts_model, body, TTS_TIMEOUT_S)
        if not j or j.get("_retry_without_thinking"):
            return None, None
        try:
            d = j["candidates"][0]["content"]["parts"][0]["inlineData"]
        except (KeyError, IndexError):
            return None, None
        mime = d.get("mimeType", "")
        if "wav" in mime:
            out = d["data"], "audio/wav"
        elif "l16" in mime.lower() or "pcm" in mime.lower():
            m = re.search(r"rate=(\d+)", mime)
            wav = _pcm_to_wav(base64.b64decode(d["data"]), int(m.group(1)) if m else 24000)
            out = base64.b64encode(wav).decode(), "audio/wav"
        elif mime.startswith("audio/"):
            out = d["data"], mime
        else:
            return None, None
        with self._cache_lock:
            if len(self._cache) >= CACHE_MAX:
                self._cache.pop(next(iter(self._cache)))
            self._cache[(speaker, text)] = out
        return out
