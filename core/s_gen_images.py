#!/usr/bin/env python3
"""
core/s_gen_images.py

Generates synthetic "spores" and "not_spores" grayscale images.
Usage (examples):
    python core/s_gen_images.py --num_images 500 --intensity 210
"""
import argparse
import shutil
from pathlib import Path
import cv2
import numpy as np
import random
import sys

def clamp_int(v):
    return max(0, min(255, int(v)))

def generate_s_image(image_path: Path, base_intensity: int, size=(512, 512), is_spore=False):
    img = np.zeros((size[0], size[1]), dtype=np.uint8)  # Black background
    center = (size[0] // 2, size[1] // 2)
    radius = 160

    # Draw the main glowing spot
    cv2.circle(img, center, radius, (clamp_int(base_intensity),), -1)

    # Choose number of small spots; spores tend to have more variety
    n_spots = random.randint(0, 100) if is_spore else random.randint(0, 40)

    for _ in range(n_spots):
        shape_type = random.choice(["circle", "lentil", "asterisk"]) if is_spore else "circle"

        # pick random position inside a bounding box then check inside circle
        spot_x = random.randint(center[0] - (radius - 8), center[0] + (radius - 8))
        spot_y = random.randint(center[1] - (radius - 8), center[1] + (radius - 8))

        if (spot_x - center[0])**2 + (spot_y - center[1])**2 <= radius**2:
            if shape_type == "circle":
                spot_radius = random.randint(2, 18)
                spot_intensity = clamp_int(random.randint(base_intensity, base_intensity + 80))
                cv2.circle(img, (spot_x, spot_y), spot_radius, (spot_intensity,), -1)

            elif shape_type == "lentil":
                major_axis = random.randint(2, 7)
                minor_axis = random.randint(1, 2)
                angle = random.randint(0, 180)
                spot_intensity = clamp_int(random.randint(base_intensity + 5, base_intensity + 100))
                cv2.ellipse(img, (spot_x, spot_y), (major_axis, minor_axis), angle, 0, 360, (spot_intensity,), -1)

            elif shape_type == "asterisk":
                spot_intensity = clamp_int(random.randint(base_intensity + 2, base_intensity + 100))
                for angle in range(0, 180, 30):
                    x1 = int(spot_x + random.randint(2, 6) * np.cos(np.radians(angle)))
                    y1 = int(spot_y + random.randint(2, 6) * np.sin(np.radians(angle)))
                    x2 = int(spot_x - random.randint(2, 6) * np.cos(np.radians(angle)))
                    y2 = int(spot_y - random.randint(2, 6) * np.sin(np.radians(angle)))
                    cv2.line(img, (x1, y1), (x2, y2), (spot_intensity,), 1)

    cv2.imwrite(str(image_path), img)


def main(num_images: int = 500, intensity: int = 210, out_dir: str = "s_images", size=(512,512), seed: int | None = None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    out_path = Path(out_dir)
    spores_dir = out_path / "spores"
    not_spores_dir = out_path / "not_spores"

    # Remove and recreate directories safely
    if out_path.exists():
        # remove only the s_images tree to avoid dangerous globbing
        shutil.rmtree(out_path)
    spores_dir.mkdir(parents=True, exist_ok=True)
    not_spores_dir.mkdir(parents=True, exist_ok=True)

    num_images = int(num_images)
    intensity = clamp_int(intensity)

    print(f"Generating {num_images} images per class in '{out_dir}' (intensity={intensity})")
    sys.stdout.flush()

    # Generate images for "spores" with extra shapes
    for i in range(num_images):
        img_path = spores_dir / f"spores_{i:04d}.png"
        generate_s_image(img_path, base_intensity=intensity, size=size, is_spore=True)
        if (i + 1) % max(1, num_images // 10) == 0:
            print(f"  spores: {i+1}/{num_images}")
            sys.stdout.flush()

    # Generate images for "not_spores" (only circles)
    for i in range(num_images):
        img_path = not_spores_dir / f"not_spores_{i:04d}.png"
        generate_s_image(img_path, base_intensity=intensity, size=size, is_spore=False)
        if (i + 1) % max(1, num_images // 10) == 0:
            print(f"  not_spores: {i+1}/{num_images}")
            sys.stdout.flush()

    print("Done generating images.")
    return {
        "spores_dir": str(spores_dir),
        "not_spores_dir": str(not_spores_dir),
        "num_images_per_class": num_images,
        "intensity": intensity
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic spore / not_spore images.")
    parser.add_argument("--num_images", type=int, default=500, help="Number of images per class")
    parser.add_argument("--intensity", type=int, default=210, help="Base intensity for main spot (0-255)")
    parser.add_argument("--out_dir", type=str, default="s_images", help="Output directory")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (optional)")
    args = parser.parse_args()

    main(num_images=args.num_images, intensity=args.intensity, out_dir=args.out_dir, seed=args.seed)
