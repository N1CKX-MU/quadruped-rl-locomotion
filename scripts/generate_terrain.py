"""Generate rough terrain heightfield for the Go2 environment.

Creates a MuJoCo XML scene with a heightfield of Gaussian-smoothed random
bumps, suitable for testing locomotion robustness.
"""

import argparse
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter


def generate_heightfield(rows=200, cols=200, roughness=0.03, smoothing=3.0, seed=42):
    """Generate a smoothed random heightfield.

    Args:
        rows, cols: Grid resolution
        roughness: Max bump height in meters
        smoothing: Gaussian filter sigma (larger = smoother terrain)
        seed: Random seed for reproducibility
    """
    rng = np.random.RandomState(seed)
    terrain = rng.uniform(0, roughness, size=(rows, cols))
    terrain = gaussian_filter(terrain, sigma=smoothing)
    # Normalize to [0, 1] for MuJoCo hfield
    terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min() + 1e-8)
    return terrain.astype(np.float32)


def create_terrain_xml(output_path, hfield_path, terrain_size=10.0, max_height=0.05):
    """Create a MuJoCo XML scene with heightfield terrain."""
    xml = f"""<mujoco model="go2_rough_terrain">
  <include file="../mujoco_menagerie/unitree_go2/go2.xml"/>

  <asset>
    <hfield name="terrain" file="{hfield_path}"
            size="{terrain_size} {terrain_size} {max_height} 0.001"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.4 0.45 0.5"
             rgb2="0.35 0.4 0.45" width="512" height="512"/>
    <material name="terrain_mat" texture="grid" texrepeat="20 20"/>
  </asset>

  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom type="hfield" hfield="terrain" material="terrain_mat"
           pos="{terrain_size/2} 0 0" contype="1" conaffinity="1"/>
  </worldbody>
</mujoco>
"""
    with open(output_path, "w") as f:
        f.write(xml)
    print(f"Terrain XML saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate rough terrain")
    parser.add_argument("--output-xml", type=str, default="configs/terrain_rough.xml")
    parser.add_argument("--output-hfield", type=str, default="configs/terrain_data.bin")
    parser.add_argument("--roughness", type=float, default=0.03,
                        help="Max bump height in meters")
    parser.add_argument("--smoothing", type=float, default=3.0,
                        help="Gaussian smoothing sigma")
    parser.add_argument("--size", type=float, default=10.0,
                        help="Terrain size in meters")
    parser.add_argument("--max-height", type=float, default=0.05,
                        help="Max heightfield elevation in meters")
    parser.add_argument("--resolution", type=int, default=200,
                        help="Heightfield grid resolution")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_xml), exist_ok=True)

    # Generate heightfield
    terrain = generate_heightfield(
        rows=args.resolution,
        cols=args.resolution,
        roughness=args.roughness,
        smoothing=args.smoothing,
    )

    # Save as binary (MuJoCo hfield format)
    terrain.tofile(args.output_hfield)
    print(f"Heightfield saved to {args.output_hfield} "
          f"({args.resolution}x{args.resolution}, "
          f"roughness={args.roughness}m, smoothing={args.smoothing})")

    # Create XML
    # Use relative path from XML location to hfield
    hfield_rel = os.path.relpath(args.output_hfield, os.path.dirname(args.output_xml))
    create_terrain_xml(args.output_xml, hfield_rel, args.size, args.max_height)

    print(f"\nTo test: python scripts/evaluate.py --render "
          f"(after modifying xml_path in config)")


if __name__ == "__main__":
    main()
