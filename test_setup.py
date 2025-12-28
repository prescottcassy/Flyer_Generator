"""
Test script for SDXL Flyer Generator
Run this to verify your setup is working correctly
"""
import sys
import os

print("🔍 Checking dependencies...")

# Check Python version
print(f"✓ Python {sys.version}")

# Check required packages
required_packages = [
    "torch",
    "diffusers",
    "transformers",
    "gradio",
    "spaces",
    "PIL"
]

missing_packages = []

for package in required_packages:
    try:
        if package == "PIL":
            __import__("PIL")
            print(f"✓ Pillow installed")
        else:
            pkg = __import__(package)
            version = getattr(pkg, "__version__", "unknown")
            print(f"✓ {package} ({version})")
    except ImportError:
        missing_packages.append(package)
        print(f"✗ {package} not found")

if missing_packages:
    print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
    print("Install with: pip install -r requirements.txt")
    sys.exit(1)

# Check CUDA availability
import torch
if torch.cuda.is_available():
    print(f"\n✓ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("\n⚠ CUDA not available - will use CPU (slower)")

# Test model import
print("\n🧪 Testing model imports...")
try:
    from model.model import SDXLModel
    from model.utils import save_image, create_output_dir
    print("✓ Model imports successful")
except Exception as e:
    print(f"✗ Model import failed: {e}")
    sys.exit(1)

# Test Gradio app import
print("\n🧪 Testing Gradio app...")
try:
    from app import create_interface
    print("✓ Gradio app imports successful")
except Exception as e:
    print(f"✗ Gradio app import failed: {e}")
    sys.exit(1)

print("\n" + "="*50)
print("✅ All checks passed!")
print("="*50)
print("\n📋 Next steps:")
print("1. Run the Gradio app: python app.py")
print("2. Open http://localhost:7860 in your browser")
print("3. Start generating flyers!")
print("\n💡 For Hugging Face Spaces deployment:")
print("   - Push your code to a Space")
print("   - Enable GPU in Space settings")
print("   - The @spaces.GPU decorator will handle GPU allocation")
