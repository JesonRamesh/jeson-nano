# Gemma bench results (spec §7.3)

- Date: 2026-11-07 15:40
- Machine: LAPTOP-N32J83LO · NVIDIA GeForce RTX 5060 Laptop GPU · MuJoCo 3.17.0 · Ollama 0.40.15
- Network during run: ONLINE ⚠ (turn Wi-Fi off for the real numbers)
- Runs per image: 5 (warm, after one cold call) · temperature 0 · num_ctx 2048 · keep_alive -1
- Images: fire:crop, survivor:crop, decoy:crop, fire:full, survivor:full, decoy:full
- Correct = fire→{fire:true, person:false}; survivor→{fire:false, person:true}; decoy→{fire:false, person:false}

Frames and crops sent to Gemma: `sgr/perception/bench/*.png`.

- `fire`: bbox (145, 109, 178, 157) in 320×240 → crop 224×326
- `survivor`: bbox (132, 135, 200, 155) in 320×240 → crop 762×224
- `decoy`: bbox (147, 72, 173, 110) in 320×240 → crop 224×327

## Summary

| Tag | VRAM before (MiB) | VRAM after (MiB) | Δ (MiB) | Cold load (s) | p50 (s) | p95 (s) | tok/s | Correct runs | Headroom (MiB) |
|---|---|---|---|---|---|---|---|---|---|
| `gemma4:e4b` | 422 / 8151 | 5097 / 8151 | 4675 | 10.9 (wall 16.3) | 4.67 | 5.30 | 119.1 | 26/30 | 3054 |
| `gemma4:e2b` | 422 / 8151 | 3479 / 8151 | 3057 | 7.4 (wall 11.2) | 2.60 | 3.40 | 173.7 | 20/30 | 4672 |

**Suggested:** `gemma4:e4b`: 26/30 runs correct, p95 5.30 s. This is the most correct answers, then the lowest p95. The pick is yours; check the VRAM headroom (the renderer needs some too).

## `gemma4:e4b`

| Image | p50 (s) | p95 (s) | min | max | Ollama total p50 (s) | Correct | Answers (count) |
|---|---|---|---|---|---|---|---|
| fire:crop | 5.09 | 5.22 | 3.98 | 5.23 | 5.09 | 5/5 | `{"fire": true, "person": false, "confidence": 1.0, "caption": "Stylized orange and yellow flames are visible against a dark background."}` ×5 |
| survivor:crop | 4.63 | 5.66 | 4.31 | 5.91 | 4.63 | 5/5 | `{"fire": false, "person": true, "confidence": 0.95, "caption": "Figure lying on the floor in a confined space."}` ×1<br>`{"fire": false, "person": true, "confidence": 0.98, "caption": "A figure is lying on the floor in a confined space."}` ×4 |
| decoy:crop | 5.08 | 5.33 | 4.94 | 5.36 | 5.08 | 1/5 | `{"fire": false, "person": false, "confidence": 0.0, "caption": "Abstract orange rectangle on a gray background."}` ×1<br>`{"fire": true, "person": false, "confidence": 0.9, "caption": "Bright orange heat source detected against a gray background."}` ×4 |
| fire:full | 4.47 | 4.64 | 4.23 | 4.64 | 4.47 | 5/5 | `{"fire": true, "person": false, "confidence": 0.95, "caption": "Small fire visible in a dark interior room."}` ×1<br>`{"fire": true, "person": false, "confidence": 0.95, "caption": "Small fire detected in a dark interior room."}` ×4 |
| survivor:full | 4.22 | 4.35 | 4.18 | 4.35 | 4.22 | 5/5 | `{"fire": false, "person": true, "confidence": 0.95, "caption": "A person is lying on the floor in a large, empty room."}` ×5 |
| decoy:full | 4.89 | 5.11 | 4.82 | 5.16 | 4.88 | 5/5 | `{"fire": false, "person": false, "confidence": 0.8, "caption": "Dark interior hallway with a central orange light source."}` ×5 |

## `gemma4:e2b`

| Image | p50 (s) | p95 (s) | min | max | Ollama total p50 (s) | Correct | Answers (count) |
|---|---|---|---|---|---|---|---|
| fire:crop | 3.40 | 3.50 | 3.13 | 3.52 | 3.40 | 5/5 | `{"fire": true, "person": false, "confidence": 0.95, "caption": "Flames are visible in the background."}` ×5 |
| survivor:crop | 2.60 | 2.62 | 2.41 | 2.62 | 2.60 | 5/5 | `{"fire": false, "person": true, "confidence": 0.98, "caption": "A person is lying down on the floor."}` ×1<br>`{"fire": false, "person": true, "confidence": 1.0, "caption": "Person lying down on the floor."}` ×4 |
| decoy:crop | 2.54 | 2.56 | 2.50 | 2.56 | 2.54 | 0/5 | `{"fire": true, "person": false, "confidence": 0.9, "caption": "An orange shape, possibly indicating heat or damage."}` ×1<br>`{"fire": true, "person": false, "confidence": 0.9, "caption": "An orange shape, possibly indicating fire or heat."}` ×4 |
| fire:full | 2.32 | 2.43 | 2.30 | 2.45 | 2.32 | 5/5 | `{"fire": true, "person": false, "confidence": 0.98, "caption": "Visible flames are present in the interior of the room."}` ×1<br>`{"fire": true, "person": false, "confidence": 0.98, "caption": "Interior view of a room with visible flames."}` ×4 |
| survivor:full | 2.73 | 3.21 | 2.67 | 3.32 | 2.73 | 5/5 | `{"fire": false, "person": true, "confidence": 0.95, "caption": "Person found lying on the floor in the room."}` ×5 |
| decoy:full | 2.62 | 2.64 | 2.59 | 2.64 | 2.62 | 0/5 | `{"fire": true, "person": false, "confidence": 0.95, "caption": "Interior view of a dark hallway with an orange heat source."}` ×1<br>`{"fire": true, "person": false, "confidence": 0.9, "caption": "Orange object suggesting fire in a dark hallway."}` ×4 |
