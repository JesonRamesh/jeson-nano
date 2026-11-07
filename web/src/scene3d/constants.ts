// Physical sizes shared by every R3F object, in metres (sim frame: z up).
// Must match the MuJoCo world (owned by P, sgr/world/). If P changes a size,
// change it here too.

/**
 * Debris block: 1.0 × 1.0 footprint, 2.0 m tall.
 * P raised the height from the spec's 1.2 m because at 1.2 m the LiDAR and
 * the drones (flying at 1.6 m) passed over the blocks.
 * The block sits on the floor, so its centre is at z = DEBRIS_SIZE.z / 2.
 */
export const DEBRIS_SIZE = { x: 1.0, y: 1.0, z: 2.0 } as const
