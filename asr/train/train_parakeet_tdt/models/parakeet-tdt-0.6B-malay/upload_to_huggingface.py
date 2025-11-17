#!/usr/bin/env python3
"""
Upload trained Parakeet TDT 0.6B Malay model to HuggingFace Hub
"""

import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo, login
from huggingface_hub.utils import HfHubHTTPError

def main():
    # Configuration
    REPO_NAME = "parakeet-tdt-0.6b-malay"
    MODEL_DIR = Path(__file__).parent
    
    # Files to upload
    files_to_upload = {
        "full.nemo": "parakeet-tdt-0.6b-malay.nemo",  # Rename for clarity
        "10p.nemo": "parakeet-tdt-0.6b-malay-10p.nemo",  # 10% checkpoint
        "parakeet-tdt--epoch=01-step=41220-val_wer=0.2038-last.ckpt": "checkpoint-epoch01-step41220.ckpt",
        "README.md": "README.md",
        "../../config.yaml": "training_config.yaml",
    }
    
    print("=" * 70)
    print("🚀 Parakeet TDT 0.6B Malay - HuggingFace Upload")
    print("=" * 70)
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
    
    print(f"✅ HuggingFace token found")
    print()
    
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
        repo_id = f"{username}/{REPO_NAME}"
        print(f"📝 Repository: {repo_id}")
        print()
    except Exception as e:
        print(f"❌ Failed to get user info: {e}")
        sys.exit(1)
    
    # Create repository (private)
    print(f"📦 Creating private repository...")
    try:
        create_repo(
            repo_id=REPO_NAME,
            token=hf_token,
            private=True,
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
    
    # Upload files
    print("📤 Uploading files...")
    print()
    
    for source_file, target_name in files_to_upload.items():
        source_path = MODEL_DIR / source_file
        
        # Check if file exists
        if not source_path.exists():
            print(f"⚠️  Skipping {source_file} (not found)")
            continue
        
        # Get file size
        file_size = source_path.stat().st_size
        size_mb = file_size / (1024 * 1024)
        size_gb = file_size / (1024 * 1024 * 1024)
        
        if size_gb >= 1:
            size_str = f"{size_gb:.2f} GB"
        else:
            size_str = f"{size_mb:.2f} MB"
        
        print(f"  📁 {target_name} ({size_str})")
        
        try:
            api.upload_file(
                path_or_fileobj=str(source_path),
                path_in_repo=target_name,
                repo_id=repo_id,
                token=hf_token,
                commit_message=f"Upload {target_name}"
            )
            print(f"     ✅ Uploaded successfully")
        except Exception as e:
            print(f"     ❌ Upload failed: {e}")
            continue
        
        print()
    
    # Final summary
    print("=" * 70)
    print("✅ Upload complete!")
    print("=" * 70)
    print()
    print(f"🔗 Repository URL: https://huggingface.co/{repo_id}")
    print()
    print("📝 Next steps:")
    print("   1. Visit the repository to verify files")
    print("   2. Check the model card (README.md)")
    print("   3. Test loading the model")
    print()
    print("💡 To load the model:")
    print(f"   from nemo.collections.asr.models import ASRModel")
    print(f"   model = ASRModel.from_pretrained('{repo_id}')")
    print()

if __name__ == "__main__":
    main()

