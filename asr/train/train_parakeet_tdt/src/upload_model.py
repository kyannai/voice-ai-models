#!/usr/bin/env python3
"""
Upload trained Parakeet TDT models to HuggingFace Hub

Supports:
- Uploading a single .nemo model file
- Uploading multiple model files (stage1, stage2, checkpoints)
- Creating model cards and documentation

Usage:
    # Upload a single model
    python src/upload_model.py --model models/full.nemo --repo-name parakeet-tdt-0.6b-malay

    # Upload with custom name in repo
    python src/upload_model.py --model models/full.nemo --repo-name parakeet-tdt-0.6b-malay --filename model.nemo

    # Upload stage2 model
    python src/upload_model.py --model models/stage2.nemo --repo-name parakeet-tdt-0.6b-malay --filename parakeet-tdt-0.6b-malay-stage2.nemo
"""

import os
import sys
import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo, login
from huggingface_hub.utils import HfHubHTTPError


def get_file_size_str(file_path: Path) -> str:
    """Get human-readable file size"""
    file_size = file_path.stat().st_size
    size_gb = file_size / (1024 * 1024 * 1024)
    size_mb = file_size / (1024 * 1024)
    
    if size_gb >= 1:
        return f"{size_gb:.2f} GB"
    else:
        return f"{size_mb:.2f} MB"


def upload_model(
    model_path: str,
    repo_name: str,
    filename: str = None,
    private: bool = True,
    commit_message: str = None
):
    """
    Upload a model to HuggingFace Hub
    
    Args:
        model_path: Path to the .nemo model file
        repo_name: Repository name (without username)
        filename: Target filename in repo (default: use source filename)
        private: Whether to create a private repo
        commit_message: Custom commit message
    """
    model_path = Path(model_path)
    
    # Validate model exists
    if not model_path.exists():
        print(f"❌ Error: Model not found at {model_path}")
        sys.exit(1)
    
    # Get target filename
    if filename is None:
        filename = model_path.name
    
    print("=" * 70)
    print("🚀 Parakeet TDT Model - HuggingFace Upload")
    print("=" * 70)
    print()
    print(f"📁 Model: {model_path}")
    print(f"📦 Size: {get_file_size_str(model_path)}")
    print(f"📝 Target: {filename}")
    print()
    
    # Check for HF_TOKEN
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("❌ Error: HF_TOKEN environment variable not found!")
        print()
        print("Please set your HuggingFace token:")
        print("  export HF_TOKEN='your_token_here'")
        print()
        print("Or login interactively:")
        print("  huggingface-cli login")
        sys.exit(1)
    
    print("✅ HuggingFace token found")
    
    # Initialize HF API
    api = HfApi()
    
    # Login
    try:
        login(token=hf_token)
        print("✅ Logged in to HuggingFace")
    except Exception as e:
        print(f"❌ Login failed: {e}")
        sys.exit(1)
    
    # Get username
    try:
        user_info = api.whoami(token=hf_token)
        username = user_info['name']
        repo_id = f"{username}/{repo_name}"
        print(f"📝 Repository: {repo_id}")
        print()
    except Exception as e:
        print(f"❌ Failed to get user info: {e}")
        sys.exit(1)
    
    # Create repository
    privacy_str = "private" if private else "public"
    print(f"📦 Creating {privacy_str} repository...")
    try:
        create_repo(
            repo_id=repo_name,
            token=hf_token,
            private=private,
            repo_type="model",
            exist_ok=True
        )
        print(f"✅ Repository created/exists: https://huggingface.co/{repo_id}")
        print()
    except HfHubHTTPError as e:
        if "already exists" in str(e):
            print(f"✅ Repository already exists: https://huggingface.co/{repo_id}")
            print()
        else:
            print(f"❌ Failed to create repository: {e}")
            sys.exit(1)
    
    # Upload file
    print(f"📤 Uploading {filename}...")
    print("This may take several minutes for large models...")
    print()
    
    if commit_message is None:
        commit_message = f"Upload {filename}"
    
    try:
        api.upload_file(
            path_or_fileobj=str(model_path),
            path_in_repo=filename,
            repo_id=repo_id,
            token=hf_token,
            commit_message=commit_message
        )
        print(f"✅ Upload successful!")
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        sys.exit(1)
    
    # Final summary
    print()
    print("=" * 70)
    print("✅ Upload complete!")
    print("=" * 70)
    print()
    print(f"🔗 Repository URL: https://huggingface.co/{repo_id}")
    print()
    print("💡 To load the model:")
    print(f"   from nemo.collections.asr.models import ASRModel")
    print(f"   model = ASRModel.from_pretrained('{repo_id}')")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Upload Parakeet TDT models to HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload main model
  python src/upload_model.py --model models/full.nemo --repo-name parakeet-tdt-0.6b-malay
  
  # Upload with custom filename
  python src/upload_model.py --model models/full.nemo --repo-name parakeet-tdt-0.6b-malay \\
      --filename parakeet-tdt-0.6b-malay.nemo
  
  # Upload stage2 model
  python src/upload_model.py --model models/stage2.nemo --repo-name parakeet-tdt-0.6b-malay \\
      --filename parakeet-tdt-0.6b-malay-stage2.nemo
  
  # Create public repository
  python src/upload_model.py --model models/full.nemo --repo-name my-model --public
        """
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Path to the .nemo model file"
    )
    parser.add_argument(
        "--repo-name", "-r",
        type=str,
        required=True,
        help="Repository name (without username prefix)"
    )
    parser.add_argument(
        "--filename", "-f",
        type=str,
        default=None,
        help="Target filename in repository (default: use source filename)"
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Create a public repository (default: private)"
    )
    parser.add_argument(
        "--message", "-msg",
        type=str,
        default=None,
        help="Custom commit message"
    )
    
    args = parser.parse_args()
    
    upload_model(
        model_path=args.model,
        repo_name=args.repo_name,
        filename=args.filename,
        private=not args.public,
        commit_message=args.message
    )


if __name__ == "__main__":
    main()
