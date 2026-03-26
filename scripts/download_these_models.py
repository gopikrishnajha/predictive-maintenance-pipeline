#!/usr/bin/env python3
"""
Download and export multiple LLMs to OpenVINO IR format for openvino-genai.
Reads model list from a file and exports each model with appropriate quantization settings.
"""

import argparse
import sys
import subprocess
from pathlib import Path
import os


# Model size classification: models up to 4B use GQ (group size 128), larger models use CW
MODEL_SIZE_CONFIG = {
    # 2B-3B models - use GQ
    "google/gemma-2-2b-it": "gq",
    "meta-llama/Llama-3.2-3B-Instruct": "gq",
    
    # 4B models - use GQ (up to 4B)
    "Qwen/Qwen3-4B": "gq",
    "zai-org/glm-edge-4b-chat": "gq",
    "microsoft/Phi-4-mini-instruct": "gq",
    "microsoft/Phi-4-mini-reasoning": "gq",
    
    # 7B-8B models - use CW
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": "cw",
    "mistralai/Mistral-7B-Instruct-v0.2": "cw",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": "cw",
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "cw",
    "Qwen/Qwen3-8B": "cw",
    
    # Larger models - use CW
    "microsoft/Phi-4-reasoning": "cw",
}


def get_model_name_from_hf_id(hf_id: str, quantization: str) -> str:
    """Convert HuggingFace model ID to local directory name."""
    model_name = hf_id.split("/")[-1]
    return f"{model_name}-int4{quantization}"


def download_and_export(model_id: str, base_output_dir: str, hf_token: str = None):
    """Download HuggingFace model and export to OpenVINO format using optimum-cli."""
    
    # Determine quantization strategy based on model size
    quantization = MODEL_SIZE_CONFIG.get(model_id, "cw")  # Default to CW if unknown
    
    # Build output directory path
    model_dir_name = get_model_name_from_hf_id(model_id, quantization)
    output_path = Path(base_output_dir) / model_dir_name
    
    # Check if model already exists
    if output_path.exists():
        print(f"✅ {model_id} is already available at {output_path}, skipping download.")
        return True
    
    print(f"\n{'='*80}")
    print(f"[Download] 🔽 Model: {model_id}")
    print(f"[Strategy] 🔧 Quantization: INT4-{'GQ-128' if quantization == 'gq' else 'CW'}")
    print(f"[Output] 📁 Saving to: {output_path}")
    print(f"{'='*80}\n")
    
    try:
        # Build optimum-cli command based on quantization strategy
        if quantization == "gq":
            # Group-wise quantization with group size 128
            cmd = [
                "optimum-cli", "export", "openvino",
                "-m", model_id,
                "--weight-format", "int4",
                "--sym",
                "--ratio", "1.0",
                "--group-size", "128",
                str(output_path)
            ]
        else:
            # Channel-wise quantization
            cmd = [
                "optimum-cli", "export", "openvino",
                "-m", model_id,
                "--weight-format", "int4",
                "--sym",
                "--ratio", "1.0",
                "--group-size", "-1",
                str(output_path)
            ]
        
        print(f"[Command] Running: {' '.join(cmd)}\n")
        
        # Set up environment with HF token if provided
        env = os.environ.copy()
        if hf_token:
            env['HF_TOKEN'] = hf_token
            env['HUGGING_FACE_HUB_TOKEN'] = hf_token
        
        # Stream output in real-time
        subprocess.run(cmd, env=env, check=True)
        
        print(f"\n✅ Model exported successfully to {output_path}")
        return True
        
    except FileNotFoundError:
        print("❌ Error: optimum-cli not found. Install with:")
        print("   pip install optimum-intel[openvino,nncf]")
        return False
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during download/export of {model_id}")
        print(f"   Return code: {e.returncode}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def read_model_list(file_path: str) -> list:
    """Read model list from a text file with [download] and [supported-models] sections.
    
    Only models under the [download] section are returned.
    Models under [supported-models] are listed but not downloaded.
    If no sections are present, all non-comment/non-empty lines are returned (backward compatible).
    """
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"❌ Error: Model list file not found: {file_path}")
        sys.exit(1)
    
    # Check if the file uses sections
    has_sections = any(line.strip().lower() in ('[download]', '[supported-models]') for line in lines)
    
    if not has_sections:
        # Backward compatible: return all non-comment, non-empty lines
        return [line.strip() for line in lines if line.strip() and not line.startswith('#')]
    
    # Parse sections
    current_section = None
    download_models = []
    supported_models = []
    
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        if stripped.lower() == '[download]':
            current_section = 'download'
            continue
        elif stripped.lower() == '[supported-models]':
            current_section = 'supported-models'
            continue
        
        if current_section == 'download':
            download_models.append(stripped)
        elif current_section == 'supported-models':
            supported_models.append(stripped)
    
    if supported_models:
        print(f"ℹ️  {len(supported_models)} supported models available (move to [download] section to install)")
    
    return download_models


def main():
    parser = argparse.ArgumentParser(
        description="Download and export multiple LLMs to OpenVINO format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all models from default list
  python scripts/download_these_models.py
  
  # Download models from custom list
  python scripts/download_these_models.py --model-list my_models.txt
  
  # Download gated models with HF token
  python scripts/download_these_models.py --token YOUR_HF_TOKEN
  
  # Or set environment variable
  export HF_TOKEN=YOUR_HF_TOKEN
  python scripts/download_these_models.py
        """
    )
    parser.add_argument(
        "--model-list",
        type=str,
        default="setup/model_list.txt",
        help="Path to file containing model IDs (one per line)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/ov_models/llms",
        help="Base output directory for exported models"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace token for gated models (or set HF_TOKEN env var)"
    )
    
    args = parser.parse_args()
    
    # Get HF token from args or environment
    hf_token = args.token or os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
    
    # Ensure we're in the right conda environment
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', '')
    if conda_env != 'pace':
        print("⚠️  Warning: Not in 'pace' conda environment")
        print("   Run: conda activate pace")
        print()
    
    # Check for HF token
    if hf_token:
        print("🔐 Using HuggingFace token for authentication")
    else:
        print("⚠️  No HuggingFace token provided")
        print("   Some models (Meta Llama, DeepSeek) may require authentication")
        print("   Use: --token YOUR_HF_TOKEN or export HF_TOKEN=YOUR_HF_TOKEN")
        print()
    
    # Read model list
    models = read_model_list(args.model_list)
    
    print(f"\n📋 Found {len(models)} models to process:")
    for i, model in enumerate(models, 1):
        quant = MODEL_SIZE_CONFIG.get(model, "cw")
        print(f"   {i:2d}. {model:50s} [INT4-{'GQ' if quant == 'gq' else 'CW'}]")
    
    # Process each model
    success_count = 0
    failed_models = []
    
    for i, model_id in enumerate(models, 1):
        print(f"\n\n{'#'*80}")
        print(f"# Processing model {i}/{len(models)}")
        print(f"{'#'*80}")
        
        if download_and_export(model_id, args.output, hf_token):
            success_count += 1
        else:
            failed_models.append(model_id)
    
    # Summary
    print(f"\n\n{'='*80}")
    print(f"📊 SUMMARY")
    print(f"{'='*80}")
    print(f"✅ Successfully exported: {success_count}/{len(models)}")
    if failed_models:
        print(f"❌ Failed models: {len(failed_models)}")
        for model in failed_models:
            print(f"   - {model}")
    else:
        print("🎉 All models exported successfully!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
