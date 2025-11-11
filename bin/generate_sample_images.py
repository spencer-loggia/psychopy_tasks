"""
Generate a small set of sample PNG images for quick testing.
Usage:
python bin/generate_sample_images.py --out_dir ./sample_images --num 6 --size 512 512
"""
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def make_color_image(path: Path, size=(512, 512), color=(200, 30, 30), text: str = ""):
    im = Image.new("RGBA", size, color + (255,))
    draw = ImageDraw.Draw(im)
    # draw a simple circle or label
    radius = min(size) // 4
    cx, cy = size[0] // 2, size[1] // 2
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=(255, 255, 255, 200))
    if text:
        try:
            font = ImageFont.load_default()
            draw.text((10, 10), text, fill=(255, 255, 255, 255), font=font)
        except Exception:
            pass
    im.save(path, "PNG")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True, help="Directory to save sample images")
    parser.add_argument("--num", type=int, default=6, help="Number of sample images to create")
    parser.add_argument("--size", type=int, nargs=2, default=(512, 512), help="Image size width height")
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    base_colors = [
        (200, 30, 30),
        (30, 200, 30),
        (30, 30, 200),
        (200, 200, 30),
        (200, 30, 200),
        (30, 200, 200),
    ]
    for i in range(args.num):
        color = base_colors[i % len(base_colors)]
        name = f"sample_{i+1:02d}.png"
        path = out / name
        make_color_image(path, size=tuple(args.size), color=color, text=name)
    print(f"Wrote {args.num} images to {out.resolve()}")


if __name__ == "__main__":
    main()
