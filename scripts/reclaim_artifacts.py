#!/usr/bin/env python3
"""
Reclaim Artifacts Script

This script extracts artifacts.zip and places the contents in their appropriate
locations within the PACE project structure.

Usage:
    python scripts/reclaim_artifacts.py [--artifacts-path PATH]

Arguments:
    --artifacts-path: Path to artifacts.zip (default: ./artifacts.zip)
"""

import argparse
import shutil
import sys
import zipfile
from pathlib import Path


def reclaim_artifacts(artifacts_path: Path, workspace_root: Path):
    """
    Extract artifacts.zip and place contents in appropriate locations.
    
    Args:
        artifacts_path: Path to artifacts.zip file
        workspace_root: Root directory of the workspace
    """
    print("🔍 PACE Artifacts Reclamation")
    print("=" * 60)
    
    # Check if artifacts file exists
    if not artifacts_path.exists():
        print(f"❌ Error: artifacts.zip not found at {artifacts_path}")
        sys.exit(1)
    
    print(f"📦 Found artifacts: {artifacts_path}")
    print(f"   Size: {artifacts_path.stat().st_size / (1024*1024):.2f} MB")
    print()
    
    # Create extraction directory
    extract_dir = workspace_root / "tmp_extract"
    if extract_dir.exists():
        print(f"🗑️  Removing old extraction directory...")
        shutil.rmtree(extract_dir)
    
    extract_dir.mkdir(exist_ok=True)
    
    # Extract artifacts
    print(f"📂 Extracting artifacts to temporary location...")
    try:
        with zipfile.ZipFile(artifacts_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"✅ Extraction complete")
    except Exception as e:
        print(f"❌ Error extracting artifacts: {e}")
        shutil.rmtree(extract_dir, ignore_errors=True)
        sys.exit(1)
    
    print()
    
    # Move datasets
    datasets_src = extract_dir / "datasets"
    datasets_dst = workspace_root / "datasets"
    
    if datasets_src.exists():
        print(f"📁 Processing datasets/")
        if datasets_dst.exists():
            print(f"   ⚠️  Target datasets/ already exists")
            response = input("   Overwrite? [y/N]: ").strip().lower()
            if response == 'y':
                print(f"   🗑️  Removing existing datasets/")
                shutil.rmtree(datasets_dst)
            else:
                print(f"   ⏭️  Skipping datasets/")
                datasets_src = None
        
        if datasets_src:
            print(f"   📋 Moving datasets/ to workspace root...")
            shutil.move(str(datasets_src), str(datasets_dst))
            print(f"   ✅ datasets/ placed successfully")
    else:
        print(f"⚠️  No datasets/ found in artifacts")
    
    print()
    
    # Move pt_models
    pt_models_src = extract_dir / "models" / "pt_models"
    pt_models_dst = workspace_root / "models" / "pt_models"
    
    if pt_models_src.exists():
        print(f"📁 Processing models/pt_models/")
        if pt_models_dst.exists():
            print(f"   ⚠️  Target models/pt_models/ already exists")
            response = input("   Overwrite? [y/N]: ").strip().lower()
            if response == 'y':
                print(f"   🗑️  Removing existing models/pt_models/")
                shutil.rmtree(pt_models_dst)
            else:
                print(f"   ⏭️  Skipping models/pt_models/")
                pt_models_src = None
        
        if pt_models_src:
            # Ensure models/ directory exists
            pt_models_dst.parent.mkdir(parents=True, exist_ok=True)
            print(f"   📋 Moving models/pt_models/ to workspace...")
            shutil.move(str(pt_models_src), str(pt_models_dst))
            print(f"   ✅ models/pt_models/ placed successfully")
    else:
        print(f"⚠️  No models/pt_models/ found in artifacts")
    
    print()
    
    # Clean up
    print(f"🧹 Cleaning up temporary files...")
    shutil.rmtree(extract_dir, ignore_errors=True)
    print(f"✅ Cleanup complete")
    
    print()
    print("=" * 60)
    print("🎉 Artifacts reclamation complete!")
    print()
    
    # Show summary
    print("📊 Summary:")
    if (workspace_root / "datasets").exists():
        dataset_count = sum(1 for _ in (workspace_root / "datasets").rglob("*") if _.is_file())
        print(f"   ✓ datasets/ ({dataset_count} files)")
    if (workspace_root / "models" / "pt_models").exists():
        pt_model_count = sum(1 for _ in (workspace_root / "models" / "pt_models").rglob("*") if _.is_file())
        print(f"   ✓ models/pt_models/ ({pt_model_count} files)")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Extract and reclaim PACE artifacts from artifacts.zip",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--artifacts-path',
        type=Path,
        default=Path('artifacts.zip'),
        help='Path to artifacts.zip file (default: ./artifacts.zip)'
    )
    
    parser.add_argument(
        '--workspace-root',
        type=Path,
        default=Path.cwd(),
        help='Workspace root directory (default: current directory)'
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    artifacts_path = args.artifacts_path.resolve()
    workspace_root = args.workspace_root.resolve()
    
    # Validate workspace root
    if not (workspace_root / "setup").exists() and not (workspace_root / "src").exists():
        print(f"⚠️  Warning: {workspace_root} may not be a valid PACE workspace")
        response = input("Continue anyway? [y/N]: ").strip().lower()
        if response != 'y':
            print("Aborted.")
            sys.exit(0)
    
    reclaim_artifacts(artifacts_path, workspace_root)


if __name__ == "__main__":
    main()
