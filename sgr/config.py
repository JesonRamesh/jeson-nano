"""Global constants (owned by S). Change values here, not inline."""

# Timing (§3)
CONTROL_DT = 0.1  # s of sim time per control tick
PHYSICS_DT = 0.005  # s per World.step substep
SUBSTEPS = 20  # World.step(SUBSTEPS) per control tick
STATE_HZ = 10.0  # wall-clock rate of `state` messages
CAMERA_FPS = 10.0  # wall-clock fps per drone camera (render pacing, §3)

# Sensors (§4, §7.1)
LIDAR_RAYS = 36  # ray k at yaw + k * 10 deg
LIDAR_MAX = 6.0  # metres; 6.0 = no hit
CAM_W, CAM_H = 320, 240
CAM_FOVY_DEG = 90.0
CAM_PITCH_DEG = -15.0

# Maps (§3)
GRID_RES = 0.1  # grid.png metres per pixel
BELIEF_RES = 0.2  # swarm belief grid

# Drones
MAX_SPEED = 1.2  # m/s (§8.2)
DRONE_RADIUS = 0.25
DRONE_COLOURS = {"D1": "#3373ff", "D2": "#1acce6", "D3": "#cc40d9", "D4": "#ebebf5"}  # = MuJoCo camera colours (P); avoid fire hues
HEARTBEAT_TIMEOUT = 3.0  # s (§7.4)

# Firefighter (§9)
FF_SPEED = 1.4  # m/s

# Gemma (§7.3) - P picks the tag in H0-H1 and writes it here
OLLAMA_URL = "http://localhost:11434/api/chat"
GEMMA_MODEL_TAG = "gemma4:e4b"  # hour-1 bench: 26/30 correct
GEMMA_TIMEOUT_S = 8.0  # measured p95 6.3 s on the demo laptop
