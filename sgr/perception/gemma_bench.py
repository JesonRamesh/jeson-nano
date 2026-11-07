# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["mujoco==3.17.0", "numpy", "pillow", "httpx"]
# ///
"""Gemma hour-1 bench (spec §7.3).

Renders the three §7.1 targets (fire, survivor, decoy) in a tiny MuJoCo scene,
crops each like the live pipeline will, and asks every Ollama model tag given on
the command line the fixed §7.3 prompt. Writes sgr/perception/bench/bench_results.md
plus the frames/crops it sent.

    uv run sgr/perception/gemma_bench.py gemma4:e4b gemma4:e2b
    uv run sgr/perception/gemma_bench.py --frames-only        # render + save, no Ollama

Run with Wi-Fi OFF on the demo laptop for the numbers that count.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import mujoco  # noqa: E402
from PIL import Image  # noqa: E402

from sgr.world import shapes  # noqa: E402

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
# what a correct answer looks like for each target
EXPECTED = {
    "fire": {"fire": True, "person": False},
    "survivor": {"fire": False, "person": True},
    "decoy": {"fire": False, "person": False},
}
CAM_DIST = 3.0
CAM_Z = 1.6
MIN_CROP = 224

OUT_DIR = Path(__file__).resolve().parent / "bench"


# ------------------------------------------------------------------ scene
def build_scene() -> str:
    """Three stations 30 m apart along x. Camera 3 m from each target, looking +x."""
    sx = {"fire": 0.0, "survivor": 30.0, "decoy": 60.0}
    wall = shapes.rgba(shapes.WALL_RGBA)
    wh = shapes.WALL_HEIGHT / 2
    bodies, cams = [], []
    for name, x in sx.items():
        # grey back wall 0.5 m behind the target (the decoy is mounted on it)
        back_x = x + 0.6
        bodies.append(
            f'<geom name="wall_{name}" type="box" pos="{back_x} 0 {wh}" size="0.1 4 {wh}" '
            f'rgba="{wall}" material="mat_matte" group="0"/>'
        )
        # side walls, like a corridor
        for side in (-1, 1):
            bodies.append(
                f'<geom name="side_{name}_{side}" type="box" pos="{x - 2} {side * 2.0} {wh}" '
                f'size="3 0.1 {wh}" rgba="{wall}" material="mat_matte" group="0"/>'
            )
        if name == "fire":
            bodies.append(shapes.fire_cluster("target_fire", x, 0.0))
            tx, tz = x, 0.5
        elif name == "survivor":
            bodies.append(shapes.survivor("target_survivor", x, 0.0, yaw=np.pi / 2 + 0.3))
            tx, tz = x, 0.15
        else:
            face_x = back_x - 0.1 - 0.011  # flush with the wall face, pointing -x
            bodies.append(shapes.decoy_panel("target_decoy", face_x, 0.0, normal_yaw=np.pi))
            tx, tz = face_x, shapes.DECOY_CENTRE_Z
        cams.append(
            f'<camera name="cam_{name}" pos="{tx - CAM_DIST} 0 {CAM_Z}" '
            f'fovy="{shapes.CAM_FOVY_DEG}" xyaxes="{shapes.camera_xyaxes(0.0)}"/>'
        )
    floor = shapes.rgba(shapes.FLOOR_RGBA)
    return f"""<mujoco model="gemma_bench">
{shapes.visual_settings()}
<asset>
{shapes.assets()}
</asset>
<worldbody>
<light directional="true" pos="0 0 10" dir="0.2 0.1 -1" diffuse="0.55 0.55 0.55" castshadow="false"/>
<geom name="floor" type="plane" size="100 100 0.1" rgba="{floor}" material="mat_matte" group="0"/>
{chr(10).join(bodies)}
{chr(10).join(cams)}
</worldbody>
</mujoco>"""


@dataclass
class Shot:
    name: str
    frame: np.ndarray  # (240, 320, 3) uint8
    bbox: tuple[int, int, int, int]  # x0, y0, x1, y1 (exclusive)
    crop: np.ndarray


def crop_for_gemma(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """§7.3: bbox padded 2x, clamped to the frame, upscaled so the short side >= 224 px."""
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    bw, bh = (x1 - x0) * 2, (y1 - y0) * 2
    nx0, nx1 = int(max(0, cx - bw / 2)), int(min(w, cx + bw / 2))
    ny0, ny1 = int(max(0, cy - bh / 2)), int(min(h, cy + bh / 2))
    img = Image.fromarray(frame[ny0:ny1, nx0:nx1])
    short = min(img.size)
    if short < MIN_CROP:
        s = MIN_CROP / short
        img = img.resize((round(img.size[0] * s), round(img.size[1] * s)), Image.LANCZOS)
    return np.asarray(img)


def render_shots() -> list[Shot]:
    model = mujoco.MjModel.from_xml_string(build_scene())
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, shapes.CAM_H, shapes.CAM_W)
    shots = []
    for name in EXPECTED:
        cam = f"cam_{name}"
        renderer.disable_segmentation_rendering()
        renderer.update_scene(data, camera=cam)
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        frame = renderer.render().copy()

        renderer.enable_segmentation_rendering()
        renderer.update_scene(data, camera=cam)
        seg = renderer.render()
        body = model.body(f"target_{name}").id
        geoms = np.flatnonzero(model.geom_bodyid == body)
        mask = (seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM) & np.isin(seg[..., 0], geoms)
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            raise RuntimeError(f"target {name} not visible from its camera")
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        shots.append(Shot(name, frame, bbox, crop_for_gemma(frame, bbox)))
    renderer.close()
    return shots


# ------------------------------------------------------------------ system probes
def vram_mib() -> tuple[int, int] | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip().splitlines()[0]
        used, total = (int(v) for v in out.split(","))
        return used, total
    except Exception:
        return None


def gpu_name() -> str:
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip().splitlines()[0]
    except Exception:
        return "unknown (no nvidia-smi)"


def network_online() -> bool:
    import socket

    try:
        with socket.create_connection(("1.1.1.1", 443), timeout=1.0):
            return True
    except OSError:
        return False


# ------------------------------------------------------------------ ollama
def png_b64(img: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@dataclass
class Call:
    wall_s: float
    ok: bool
    answer: dict | None = None
    raw: str = ""
    error: str = ""
    total_s: float = 0.0
    load_s: float = 0.0
    eval_count: int = 0
    eval_s: float = 0.0


def ask(client, host: str, tag: str, img_b64: str, timeout: float) -> Call:
    body = {
        "model": tag,
        "stream": False,
        "keep_alive": -1,
        "format": SCHEMA,
        "options": {"temperature": 0, "num_ctx": 2048},
        "messages": [{"role": "user", "content": PROMPT, "images": [img_b64]}],
    }
    t0 = time.perf_counter()
    try:
        r = client.post(f"{host}/api/chat", json=body, timeout=timeout)
        wall = time.perf_counter() - t0
        r.raise_for_status()
        j = r.json()
    except Exception as e:  # recorded, never fatal
        return Call(time.perf_counter() - t0, False, error=f"{type(e).__name__}: {e}"[:300])
    raw = j.get("message", {}).get("content", "")
    call = Call(
        wall, True, raw=raw,
        total_s=j.get("total_duration", 0) / 1e9,
        load_s=j.get("load_duration", 0) / 1e9,
        eval_count=j.get("eval_count", 0),
        eval_s=j.get("eval_duration", 0) / 1e9,
    )
    try:
        call.answer = json.loads(raw)
    except json.JSONDecodeError:
        call.ok, call.error = False, "invalid JSON"
    return call


def unload_all(client, host: str) -> None:
    try:
        for m in client.get(f"{host}/api/ps", timeout=10).json().get("models", []):
            client.post(f"{host}/api/generate", json={"model": m["name"], "keep_alive": 0}, timeout=30)
        for _ in range(40):  # wait until VRAM is actually released
            if not client.get(f"{host}/api/ps", timeout=10).json().get("models"):
                break
            time.sleep(0.25)
        time.sleep(1.0)
    except Exception:
        pass


@dataclass
class TagResult:
    tag: str
    vram_before: tuple[int, int] | None = None
    vram_after: tuple[int, int] | None = None
    cold: Call | None = None
    calls: dict[str, list[Call]] = field(default_factory=dict)  # image key -> calls


def bench_tag(client, host, tag, images: dict[str, np.ndarray], runs: int, timeout: float) -> TagResult:
    res = TagResult(tag)
    unload_all(client, host)
    res.vram_before = vram_mib()
    print(f"[{tag}] cold call (loads the model)...", flush=True)
    first = next(iter(images.values()))
    res.cold = ask(client, host, tag, png_b64(first), timeout)
    if not res.cold.ok:
        print(f"[{tag}] cold call failed: {res.cold.error}", flush=True)
    for key, img in images.items():
        b64 = png_b64(img)
        res.calls[key] = []
        for i in range(runs):
            c = ask(client, host, tag, b64, timeout)
            res.calls[key].append(c)
            status = f"{c.wall_s:.2f}s" if c.ok else f"FAIL {c.error}"
            print(f"[{tag}] {key} run {i + 1}/{runs}: {status}", flush=True)
    res.vram_after = vram_mib()
    unload_all(client, host)
    return res


# ------------------------------------------------------------------ report
def target_of(key: str) -> str:
    return key.split(":")[0]


def correct(answer: dict | None, target: str) -> bool:
    if not answer:
        return False
    exp = EXPECTED[target]
    return all(answer.get(k) is v for k, v in exp.items())


def pct(xs: list[float], q: float) -> float:
    return float(np.percentile(xs, q)) if xs else float("nan")


def fmt_vram(v) -> str:
    return f"{v[0]} / {v[1]}" if v else "n/a"


def summarise(r: TagResult) -> dict:
    lat = [c.wall_s for cs in r.calls.values() for c in cs if c.ok]
    runs = [(c, target_of(k)) for k, cs in r.calls.items() for c in cs]
    n_correct = sum(correct(c.answer, t) for c, t in runs if c.ok)
    toks = [c.eval_count / c.eval_s for cs in r.calls.values() for c in cs if c.ok and c.eval_s > 0]
    return {
        "p50": pct(lat, 50), "p95": pct(lat, 95), "n_ok": len(lat),
        "n_correct": n_correct, "n_runs": len(runs),
        "tok_s": float(np.mean(toks)) if toks else float("nan"),
        "headroom": (r.vram_after[1] - r.vram_after[0]) if r.vram_after else None,
    }


def write_report(path: Path, shots: list[Shot], results: list[TagResult], args, online: bool, ollama_ver: str):
    L = []
    L.append("# Gemma bench results (spec §7.3)\n")
    L.append(f"- Date: {datetime.now():%Y-%m-%d %H:%M}")
    L.append(f"- Machine: {platform.node()} · {gpu_name()} · MuJoCo {mujoco.__version__} · Ollama {ollama_ver}")
    L.append(f"- Network during run: {'ONLINE ⚠ (turn Wi-Fi off for the real numbers)' if online else 'OFFLINE ✓'}")
    L.append(f"- Runs per image: {args.runs} (warm, after one cold call) · temperature 0 · num_ctx 2048 · keep_alive -1")
    L.append(f"- Images: {', '.join(k for k in results[0].calls) if results else '-'}")
    L.append("- Correct = fire→{fire:true, person:false}; survivor→{fire:false, person:true}; decoy→{fire:false, person:false}\n")
    L.append("Frames and crops sent to Gemma: `sgr/perception/bench/*.png`.\n")
    for s in shots:
        L.append(f"- `{s.name}`: bbox {s.bbox} in 320×240 → crop {s.crop.shape[1]}×{s.crop.shape[0]}")
    L.append("")

    L.append("## Summary\n")
    L.append("| Tag | VRAM before (MiB) | VRAM after (MiB) | Δ (MiB) | Cold load (s) | p50 (s) | p95 (s) | tok/s | Correct runs | Headroom (MiB) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    sums = {}
    for r in results:
        s = sums[r.tag] = summarise(r)
        delta = r.vram_after[0] - r.vram_before[0] if (r.vram_after and r.vram_before) else None
        cold = f"{r.cold.load_s:.1f} (wall {r.cold.wall_s:.1f})" if r.cold and r.cold.ok else f"FAIL: {r.cold.error if r.cold else '-'}"
        L.append(
            f"| `{r.tag}` | {fmt_vram(r.vram_before)} | {fmt_vram(r.vram_after)} | {delta if delta is not None else 'n/a'} | "
            f"{cold} | {s['p50']:.2f} | {s['p95']:.2f} | "
            f"{s['tok_s']:.1f} | {s['n_correct']}/{s['n_runs']} | {s['headroom'] if s['headroom'] is not None else 'n/a'} |"
        )
    L.append("")

    ok = [r for r in results if sums[r.tag]["n_ok"]]
    if ok:
        best = max(ok, key=lambda r: (sums[r.tag]["n_correct"], -sums[r.tag]["p95"]))
        s = sums[best.tag]
        L.append(
            f"**Suggested:** `{best.tag}`: {s['n_correct']}/{s['n_runs']} runs correct, p95 {s['p95']:.2f} s. "
            "This is the most correct answers, then the lowest p95. The pick is yours; check the VRAM headroom (the renderer needs some too).\n"
        )

    for r in results:
        L.append(f"## `{r.tag}`\n")
        L.append("| Image | p50 (s) | p95 (s) | min | max | Ollama total p50 (s) | Correct | Answers (count) |")
        L.append("|---|---|---|---|---|---|---|---|")
        for key, cs in r.calls.items():
            lat = [c.wall_s for c in cs if c.ok]
            tot = [c.total_s for c in cs if c.ok]
            counts: dict[str, int] = {}
            for c in cs:
                k = json.dumps(c.answer, ensure_ascii=False) if c.ok else f"ERROR: {c.error}"
                counts[k] = counts.get(k, 0) + 1
            answers = "<br>".join(f"`{a}` ×{n}" for a, n in counts.items()).replace("|", "\\|")
            n_ok_correct = sum(correct(c.answer, target_of(key)) for c in cs if c.ok)
            L.append(
                f"| {key} | {pct(lat, 50):.2f} | {pct(lat, 95):.2f} | {min(lat, default=float('nan')):.2f} | "
                f"{max(lat, default=float('nan')):.2f} | {pct(tot, 50):.2f} | {n_ok_correct}/{len(cs)} | {answers} |"
            )
        L.append("")
    path.write_text("\n".join(L), encoding="utf-8")


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tags", nargs="*", help="Ollama model tags, e.g. gemma4:e4b gemma4:e2b")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--timeout", type=float, default=180.0, help="per request, seconds")
    ap.add_argument("--full-frame", action="store_true", help="also send the uncropped 320x240 frames")
    ap.add_argument("--frames-only", action="store_true", help="render and save images, skip Ollama")
    ap.add_argument("--out", type=Path, default=OUT_DIR / "bench_results.md")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shots = render_shots()
    for s in shots:
        Image.fromarray(s.frame).save(OUT_DIR / f"frame_{s.name}.png")
        Image.fromarray(s.crop).save(OUT_DIR / f"crop_{s.name}.png")
    print(f"saved frames + crops to {OUT_DIR}")
    if args.frames_only:
        return 0
    if not args.tags:
        ap.error("give at least one model tag (or --frames-only)")

    import httpx

    images = {f"{s.name}:crop": s.crop for s in shots}
    if args.full_frame:
        images.update({f"{s.name}:full": s.frame for s in shots})

    online = network_online()
    if online:
        print("WARNING: network is ONLINE. Turn Wi-Fi off for the numbers that count.")
    with httpx.Client() as client:
        try:
            ver = client.get(f"{args.host}/api/version", timeout=5).json().get("version", "?")
        except Exception as e:
            print(f"Ollama not reachable at {args.host}: {e}")
            return 2
        results = [bench_tag(client, args.host, t, images, args.runs, args.timeout) for t in args.tags]
    write_report(args.out, shots, results, args, online, ver)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
