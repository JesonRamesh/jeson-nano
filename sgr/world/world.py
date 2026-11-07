"""World facade (owned by P). §4 signatures, v1.

PLACEHOLDER: until P implements MuJoCo here, World is the kinematic FakeWorld.
P: replace this class with the real implementation (keep the exact method signatures).
Do NOT delete sgr/world/fake_world.py; S develops and tests the swarm against it.

    class World:
        def __init__(self, map_dir: Path, scenario: Scenario): ...
        def step(self, n: int) -> None
        def set_drone_cmd(self, drone: str, vx: float, vy: float, vz: float, wz: float) -> None
        def drone_pose(self, drone: str) -> Pose
        def lidar(self, drone: str) -> np.ndarray                    # (36,) metres, 6.0 = no hit
        def temperature(self, drone: str) -> float
        def render_drone(self, drone: str) -> tuple[np.ndarray, CamInfo]  # 240x320x3 RGB
        def pixel_to_world(self, drone: str, u: int, v: int) -> XY | None
        def place_debris(self, x: float, y: float) -> Handle | None
        def place_fire(self, x: float, y: float) -> Handle | None
        def remove(self, h: Handle) -> None
        def kill_drone(self, drone: str) -> None
        def set_ff_input(self, fwd: float, strafe: float, yaw: float) -> None
        def ff_pose(self) -> Pose
        def push_ff_back(self) -> None
        def ground_truth(self) -> GroundTruth                        # ONLY mock perception + summary
        def get_state(self) -> bytes
        def set_state(self, b: bytes) -> None
"""

from __future__ import annotations

from sgr.world.fake_world import FakeWorld


class World(FakeWorld):
    """Placeholder: behaves exactly like FakeWorld until P lands the MuJoCo version."""
