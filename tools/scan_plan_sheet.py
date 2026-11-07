#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "opencv-python-headless"]
# ///
"""Planning sheet: the top-down scan with a metre grid, every node/edge labelled, debris,
checkpoints, entrance, survivor, fires and decoy -> maps/<name>/plan_sheet.png

    python tools/scan_plan_sheet.py maps/<name>

Use it to say exactly where things go ("fire at (24, 33)", "debris on e10").
Walls from grid.png are tinted dark red; coordinates are world metres (x east, y north).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

VIOLET = (0xE3, 0x3B, 0x40)
PEACH = (0xFF, 0x92, 0xD2)
CYAN = (0xFB, 0xFF, 0x78)
AMBER = (0x30, 0xC0, 0xFF)
RED = (0x30, 0x30, 0xE8)
GREEN = (0x60, 0xD0, 0x60)
DARK = (0x1E, 0x10, 0x0D)
WHITE = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def label(img, s, org, scale=0.8, fg=WHITE, bg=DARK, th=2):
    (tw, tht), _ = cv2.getTextSize(s, FONT, scale, th)
    x, y = org
    if x + tw + 6 > img.shape[1]:          # keep inside the image: flip to the left side
        x = max(4, x - tw - 70)
    cv2.rectangle(img, (x - 4, y - tht - 6), (x + tw + 4, y + 6), bg, -1)
    cv2.putText(img, s, (x, y), FONT, scale, fg, th, cv2.LINE_AA)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("map_dir", type=Path)
    a = ap.parse_args()
    md = a.map_dir
    sc = json.loads((md / "scenario.json").read_text())
    meta = json.loads((md / "topdown.json").read_text())
    img = cv2.imread(str(md / meta["image"]))
    res = meta["res"]
    H, W = img.shape[:2]
    grid = cv2.imread(str(md / "grid.png"), cv2.IMREAD_GRAYSCALE)
    walls = cv2.resize((grid <= 127).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
    img[walls] = (0.35 * img[walls] + 0.65 * np.array((0x2A, 0x1C, 0x6E))).astype(np.uint8)

    P = lambda x, y: (int(round(x / res)), int(round(H - y / res)))
    # metre grid: thin every 1 m, thicker + labels every 5 m
    xmax, ymax = W * res, H * res
    for v in range(0, int(xmax) + 1):
        c = (90, 90, 90) if v % 5 else (170, 170, 170)
        cv2.line(img, P(v, 0), P(v, ymax), c, 2 if v % 5 == 0 else 1)
    for v in range(0, int(ymax) + 1):
        c = (90, 90, 90) if v % 5 else (170, 170, 170)
        cv2.line(img, P(0, v), P(xmax, v), c, 2 if v % 5 == 0 else 1)
    for v in range(0, int(xmax) + 1, 5):
        label(img, f"x={v}", (P(v, 0)[0] + 4, H - 12), 0.7)
        label(img, f"x={v}", (P(v, 0)[0] + 4, 30), 0.7)
    for v in range(5, int(ymax) + 1, 5):
        label(img, f"y={v}", (8, P(0, v)[1] - 6), 0.7)
        label(img, f"y={v}", (W - 90, P(0, v)[1] - 6), 0.7)

    g = sc["graph"]
    pos = {n["id"]: (n["x"], n["y"]) for n in g["nodes"]}
    blk = {c["edge"]: c.get("at", 0.5) for c in sc.get("compromised", [])}
    for e in g["edges"]:
        pa, pb = P(*pos[e["a"]]), P(*pos[e["b"]])
        cv2.line(img, pa, pb, DARK, 9, cv2.LINE_AA)
        cv2.line(img, pa, pb, VIOLET, 5, cv2.LINE_AA)
    for e in g["edges"]:
        (xa, ya), (xb, yb) = pos[e["a"]], pos[e["b"]]
        mx, my = P((xa + xb) / 2, (ya + yb) / 2)
        label(img, e["id"], (mx + 8, my + 22), 0.6, fg=(0xF3, 0xCB, 0xCB))
        if e["id"] in blk:
            t = blk[e["id"]]
            px, py = P(xa + t * (xb - xa), ya + t * (yb - ya))
            s = 16
            cv2.line(img, (px - s, py - s), (px + s, py + s), RED, 7, cv2.LINE_AA)
            cv2.line(img, (px - s, py + s), (px + s, py - s), RED, 7, cv2.LINE_AA)
            label(img, "DEBRIS", (px + 18, py - 10), 0.6, fg=WHITE, bg=RED)
    cps = {c.get("node"): i + 1 for i, c in enumerate(sc.get("checkpoints", []))}
    for n in g["nodes"]:
        p = P(n["x"], n["y"])
        if n["id"] in cps:
            cv2.circle(img, p, 26, AMBER, 5, cv2.LINE_AA)
        col = CYAN if n["kind"] == "entrance" else PEACH
        cv2.circle(img, p, 13, col, -1, cv2.LINE_AA)
        cv2.circle(img, p, 13, DARK, 3, cv2.LINE_AA)
        txt = n["id"] + (f"  CP{cps[n['id']]}" if n["id"] in cps else "") + \
            ("  ENTRANCE" if n["kind"] == "entrance" else "")
        label(img, txt, (p[0] + 22, p[1] - 14), 1.0)
        label(img, f"({n['x']:.1f}, {n['y']:.1f})", (p[0] + 22, p[1] + 22), 0.55, fg=(200, 200, 200))
    ent = sc.get("entrance")
    if ent:
        p = P(ent["x"], ent["y"])
        q = (int(p[0] + 60 * math.cos(ent.get("yaw", 0))), int(p[1] - 60 * math.sin(ent.get("yaw", 0))))
        cv2.arrowedLine(img, p, q, CYAN, 5, cv2.LINE_AA, tipLength=0.35)
    sv = sc.get("survivor")
    if sv:
        p = P(sv["x"], sv["y"])
        cv2.drawMarker(img, p, GREEN, cv2.MARKER_STAR, 46, 5, cv2.LINE_AA)
        label(img, "SURVIVOR", (p[0] - 60, p[1] + 50), 0.8, fg=DARK, bg=GREEN)
    for f in sc.get("fires", []):
        p = P(f["x"], f["y"])
        cv2.circle(img, p, int(f.get("size", 1.5) / 2 / res), (0, 120, 255), 4, cv2.LINE_AA)
        label(img, "FIRE", (p[0] + 10, p[1]), 0.8, fg=WHITE, bg=(0, 90, 220))
    dc = sc.get("decoy")
    if dc:
        p = P(dc["x"], dc["y"])
        cv2.rectangle(img, (p[0] - 14, p[1] - 14), (p[0] + 14, p[1] + 14), (0, 140, 255), -1)
        label(img, "DECOY", (p[0] + 18, p[1]), 0.8, fg=WHITE, bg=(0, 110, 220))
    for i, (x, y) in enumerate(sc.get("drones", {}).get("start", [])):
        p = P(x, y)
        cv2.circle(img, p, 7, WHITE, -1, cv2.LINE_AA)
    # legend
    lines = [("nN (x, y)", PEACH, "node id + position (m)"), ("eN", (0xF3, 0xCB, 0xCB), "edge id"),
             ("X DEBRIS", RED, "blocked edge (drones discover it)"), ("O CPn", AMBER, "drone checkpoint"),
             ("->", CYAN, "entrance + facing"), ("*", GREEN, "survivor"), ("dark red", (0x2A, 0x1C, 0x6E), "wall in grid.png")]
    x0, y0 = W - 560, H - 40 - 38 * len(lines)
    cv2.rectangle(img, (x0 - 14, y0 - 40), (W - 14, H - 24), DARK, -1)
    for i, (k, c, d) in enumerate(lines):
        cv2.putText(img, k, (x0, y0 + 38 * i), FONT, 0.7, c, 2, cv2.LINE_AA)
        cv2.putText(img, d, (x0 + 150, y0 + 38 * i), FONT, 0.7, WHITE, 1, cv2.LINE_AA)
    out = md / "plan_sheet.png"
    cv2.imwrite(str(out), img)
    print(f"wrote {out} ({W}x{H}, 1 m grid, labels every 5 m)")


if __name__ == "__main__":
    main()
