#!/usr/bin/env python3
"""
Download a dataset from a given URL and prepare train/val splits.

Splits the dataset randomly: 90% train, 10% val.
Places data in: datasets/{use-case-id}/ (use-case-id read from config.json)

IMPORTANT DISCLAIMER:
    By using this script, you acknowledge that YOU are solely responsible for
    ensuring you have the necessary rights, permissions, and licenses to
    download and use any dataset you provide. We take no responsibility for
    any misuse of data or violation of terms of service.

Usage:
    conda activate pace
    python scripts/download_and_prep_data.py <dataset_url>

    # Example with Kaggle dataset
    python scripts/download_and_prep_data.py "https://www.kaggle.com/api/v1/datasets/download/simplexitypipeline/pipeline-defect-dataset"

    # Custom split ratio
    python scripts/download_and_prep_data.py "https://example.com/dataset.zip" --train-ratio 0.8

    # Custom output directory
    python scripts/download_and_prep_data.py "https://example.com/dataset.zip" --output datasets/my_dataset
"""

import argparse
import sys
import random
import shutil
import zipfile
import subprocess
import json
from pathlib import Path

try:
    import cv2
except ImportError:
    cv2 = None


DISCLAIMER = """
⚠️  DISCLAIMER: By using this script, you acknowledge that YOU are solely
    responsible for ensuring you have the necessary rights, permissions, and
    licenses to download and use the dataset at the provided URL. We take no
    responsibility for any misuse of data or violation of terms of service.
"""


def download_dataset(dataset_url: str, download_dir: Path) -> Path:
    """Download the dataset from the given URL using curl."""
    print("📥 Downloading dataset...")
    print(f"   URL: {dataset_url}")
    print(f"   Download dir: {download_dir}")
    print()

    download_dir.mkdir(parents=True, exist_ok=True)
    zip_path = download_dir / "pipeline-defect-dataset.zip"

    try:
        cmd = [
            "curl", "-L", "-o", str(zip_path),
            dataset_url
        ]
        print(f"   Running: {' '.join(cmd)}\n")
        subprocess.run(cmd, check=True)
        print("✅ Download complete\n")

        # Unzip
        print("📦 Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(download_dir)
        zip_path.unlink()
        print("✅ Extraction complete\n")

        return download_dir

    except FileNotFoundError:
        print("❌ Error: 'curl' not found. Please install curl.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error downloading dataset (return code: {e.returncode})")
        sys.exit(1)
    except zipfile.BadZipFile:
        print("❌ Error: Downloaded file is not a valid zip archive.")
        sys.exit(1)


def find_images_and_labels(source_dir: Path):
    """Recursively find all image files and their corresponding label files.
    
    Matches images to labels by filename stem (e.g., img001.jpg -> img001.txt),
    regardless of directory structure. This handles various dataset layouts:
    - Parallel images/labels directories
    - Flat structure with mixed files
    - Pre-split train/val/test directories
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # Step 1: Find all images
    images = []
    for f in source_dir.rglob('*'):
        if f.suffix.lower() in image_extensions and f.is_file():
            images.append(f)
    images.sort(key=lambda p: p.name)
    
    # Step 2: Find all .txt label files and index by stem
    label_map = {}
    for f in source_dir.rglob('*.txt'):
        if f.is_file():
            stem = f.stem
            # If multiple .txt files share the same stem, prefer the one
            # in a directory named 'labels'
            if stem not in label_map or 'labels' in str(f.parent).lower():
                label_map[stem] = f
    
    # Step 3: Match images to labels by stem
    pairs = []
    matched = 0
    for img_path in images:
        label_path = label_map.get(img_path.stem)
        if label_path:
            matched += 1
        pairs.append((img_path, label_path))
    
    print(f"   Matched {matched}/{len(images)} images with label files")
    
    return pairs


def split_dataset(pairs: list, train_ratio: float, seed: int):
    """Randomly split (image, label) pairs into train and val sets."""
    random.seed(seed)
    shuffled = pairs.copy()
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * train_ratio)
    train_pairs = shuffled[:split_idx]
    val_pairs = shuffled[split_idx:]

    return train_pairs, val_pairs


def copy_pairs(pairs: list, dest_images_dir: Path, dest_labels_dir: Path, split_name: str):
    """Copy image/label pairs to destination directories."""
    dest_images_dir.mkdir(parents=True, exist_ok=True)
    dest_labels_dir.mkdir(parents=True, exist_ok=True)

    print(f"   Copying {len(pairs)} samples to {split_name}...")

    for img_path, label_path in pairs:
        # Copy image
        shutil.copy2(img_path, dest_images_dir / img_path.name)
        # Copy label if it exists
        if label_path and label_path.exists():
            shutil.copy2(label_path, dest_labels_dir / label_path.name)

    print(f"   ✅ {split_name}: {len(pairs)} images copied")


def create_video_from_images(images_dir: Path, video_path: Path, fps: int = 30):
    """Create an MP4 video from all .jpg images in a directory.
    
    Args:
        images_dir: Directory containing .jpg images
        video_path: Output video file path
        fps: Frames per second (default: 30)
    """
    if cv2 is None:
        print("   ⚠️  opencv-python not installed, skipping video creation")
        print("   Install with: pip install opencv-python")
        return False
    
    image_files = sorted(images_dir.glob("*.jpg"))
    if not image_files:
        print(f"   ⚠️  No .jpg images found in {images_dir}, skipping video creation")
        return False
    
    # Read the first image to get dimensions
    first_frame = cv2.imread(str(image_files[0]))
    if first_frame is None:
        print(f"   ⚠️  Could not read {image_files[0]}, skipping video creation")
        return False
    
    h, w = first_frame.shape[:2]
    video_path.parent.mkdir(parents=True, exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))
    
    written = 0
    for img_file in image_files:
        frame = cv2.imread(str(img_file))
        if frame is not None:
            # Resize if dimensions don't match the first frame
            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h))
            writer.write(frame)
            written += 1
    
    writer.release()
    print(f"   ✅ Video created: {video_path} ({written} frames, {fps} fps, {written/fps:.1f}s)")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download a dataset and prepare train/val splits",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: 90/10 split
  python scripts/download_and_prep_data.py "https://www.kaggle.com/api/v1/datasets/download/simplexitypipeline/pipeline-defect-dataset"

  # Custom split ratio (80/20)
  python scripts/download_and_prep_data.py "https://example.com/dataset.zip" --train-ratio 0.8

  # With a specific random seed for reproducibility
  python scripts/download_and_prep_data.py "https://example.com/dataset.zip" --seed 42
        """
    )
    parser.add_argument(
        "dataset_url",
        type=str,
        help="URL to download the dataset from (e.g. a Kaggle dataset URL)"
    )
    # Read use-case-id from config.json for default output path
    config_path = Path("config.json")
    if config_path.exists():
        with open(config_path) as f:
            use_case_id = json.load(f).get("use-case-id", "pipeline_defects_detection")
    else:
        use_case_id = "pipeline_defects_detection"
    default_output = f"datasets/{use_case_id}"

    parser.add_argument(
        "--output",
        type=str,
        default=default_output,
        help=f"Output directory for the prepared dataset (default: {default_output})"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="Fraction of data for training (default: 0.9 = 90%% train, 10%% val)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits (default: 42)"
    )
    parser.add_argument(
        "--keep-download",
        action="store_true",
        help="Keep the raw downloaded files after splitting"
    )

    args = parser.parse_args()

    output_dir = Path(args.output)
    download_dir = output_dir / "_raw_download"

    # Check if dataset already exists
    if output_dir.exists() and (output_dir / "images").exists():
        print(f"✅ Dataset already available at {output_dir}, skipping download.")
        return

    print(DISCLAIMER)
    print("=" * 70)
    print("Dataset — Download & Prepare")
    print("=" * 70)
    print(f"  Output dir:    {output_dir}")
    print(f"  Train ratio:   {args.train_ratio:.0%}")
    print(f"  Val ratio:     {1 - args.train_ratio:.0%}")
    print(f"  Random seed:   {args.seed}")
    print()

    # Step 1: Download
    download_dataset(args.dataset_url, download_dir)

    # Step 2: Find all image/label pairs in the downloaded data
    print("🔍 Scanning for images and labels...")
    pairs = find_images_and_labels(download_dir)
    print(f"   Found {len(pairs)} images ({sum(1 for _, l in pairs if l)} with labels)\n")

    if not pairs:
        print("❌ No images found in downloaded data. Check the dataset structure.")
        print(f"   Downloaded to: {download_dir}")
        sys.exit(1)

    # Step 3: Split
    print(f"🔀 Splitting dataset (seed={args.seed})...")
    train_pairs, val_pairs = split_dataset(pairs, args.train_ratio, args.seed)
    print(f"   Train: {len(train_pairs)} samples ({len(train_pairs)/len(pairs):.0%})")
    print(f"   Val:   {len(val_pairs)} samples ({len(val_pairs)/len(pairs):.0%})\n")

    # Step 4: Copy to final directories
    print("📂 Organizing dataset...")
    copy_pairs(train_pairs, output_dir / "images" / "train", output_dir / "labels" / "train", "train")
    copy_pairs(val_pairs, output_dir / "images" / "val", output_dir / "labels" / "val", "val")
    print()

    # Step 4b: Create dataset.yaml config
    dataset_yaml_path = output_dir / "dataset.yaml"
    dataset_yaml_content = (
        f"path: {output_dir}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names: [Deformation, Obstacle, Rupture, Disconnect, Misalignment, Deposition]\n"
    )
    with open(dataset_yaml_path, 'w') as f:
        f.write(dataset_yaml_content)
    print(f"📋 Created {dataset_yaml_path}")
    print()

    # Step 5: Create video from val images
    print("🎬 Creating video from validation images...")
    video_path = output_dir / "video" / "input.mp4"
    val_images_dir = output_dir / "images" / "val"
    create_video_from_images(val_images_dir, video_path)
    print()

    # Step 6: Clean up raw download
    if not args.keep_download:
        print("🧹 Cleaning up raw download...")
        shutil.rmtree(download_dir, ignore_errors=True)
        print("   ✅ Raw download removed\n")

    # Summary
    print("=" * 70)
    print("✅ Dataset ready!")
    print("=" * 70)
    print(f"  📁 {output_dir}/images/train/  ({len(train_pairs)} images)")
    print(f"  📁 {output_dir}/images/val/    ({len(val_pairs)} images)")
    print(f"  📁 {output_dir}/labels/train/  (YOLO annotations)")
    print(f"  📁 {output_dir}/labels/val/    (YOLO annotations)")
    print(f"  🎬 {output_dir}/video/input.mp4  (video from val images)")
    print()
    print(f"  Dataset config: {output_dir}/dataset.yaml")
    print()


if __name__ == "__main__":
    main()
