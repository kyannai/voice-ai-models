#!/usr/bin/env python3
"""
Push Stage 2 model to Hugging Face Hub (main branch)
"""
from huggingface_hub import HfApi
from pathlib import Path

def push_stage2_model():
    """Upload Stage 2 model to main branch alongside the original"""
    
    # Configuration
    model_path = "./models/parakeet-tdt-5k-v3.nemo"
    repo_id = "kyannai/parakeet-tdt-0.6b-malay"
    filename_in_repo = "parakeet-tdt-0.6b-malay-stage2.nemo"
    
    # Verify model exists
    if not Path(model_path).exists():
        print(f"❌ Error: Model not found at {model_path}")
        return
    
    # Get file size
    size_mb = Path(model_path).stat().st_size / (1024 * 1024)
    print(f"📦 Model size: {size_mb:.1f} MB")
    
    # Initialize API
    api = HfApi()
    
    # Check if logged in
    try:
        user = api.whoami()
        print(f"✅ Logged in as: {user['name']}")
    except Exception as e:
        print("❌ Not logged in to Hugging Face.")
        print("Please run: huggingface-cli login")
        return
    
    # Upload model
    print(f"\n📤 Uploading {filename_in_repo} to {repo_id}...")
    print("This may take several minutes depending on your connection...")
    
    try:
        api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo=filename_in_repo,
            repo_id=repo_id,
            repo_type="model",
            commit_message="Add Stage 2 model: fine-tuned on Malaysian names and numbers",
            create_pr=False
        )
        
        print(f"\n✅ Model uploaded successfully!")
        print(f"\n📋 Model Information:")
        print(f"   Repository: https://huggingface.co/{repo_id}")
        print(f"   Files:")
        print(f"     - parakeet-tdt-0.6b-malay.nemo (Stage 1: original)")
        print(f"     - {filename_in_repo} (Stage 2: names & numbers)")
        print(f"\n🔗 View at: https://huggingface.co/{repo_id}")
        
        print(f"\n📖 Usage:")
        print(f"""
# Stage 1 (original model)
from huggingface_hub import hf_hub_download
import nemo.collections.asr as nemo_asr

model_path = hf_hub_download(
    repo_id="{repo_id}",
    filename="parakeet-tdt-0.6b-malay.nemo"
)
model = nemo_asr.models.ASRModel.restore_from(model_path)

# Stage 2 (names & numbers optimized)
model_path = hf_hub_download(
    repo_id="{repo_id}",
    filename="{filename_in_repo}"
)
model = nemo_asr.models.ASRModel.restore_from(model_path)
""")
        
    except Exception as e:
        print(f"\n❌ Upload failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. Verify you have write access to the repository")
        print("3. Try: huggingface-cli login --token YOUR_TOKEN")


if __name__ == "__main__":
    print("="*70)
    print("📤 Parakeet TDT Stage 2 Model Uploader")
    print("="*70)
    print("\nThis will upload your Stage 2 model as:")
    print("  parakeet-tdt-0.6b-malay-stage2.nemo")
    print("\nYour original model will remain as:")
    print("  parakeet-tdt-0.6b-malay.nemo")
    print("="*70)
    
    confirm = input("\n Continue? (y/n): ")
    if confirm.lower() == 'y':
        push_stage2_model()
    else:
        print("Cancelled.")
