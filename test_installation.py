"""
Installation Test Script

Run this script to verify that all dependencies are properly installed
and your GPU is correctly configured.
"""

import sys

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing package imports...")
    print("-" * 60)

    packages = {
        "torch_directml": "DirectML backend",
        "diffusers": "Diffusers library",
        "transformers": "Transformers library",
        "gradio": "Gradio web UI",
        "PIL": "Pillow image library",
        "numpy": "NumPy"
    }

    failed = []

    for package, description in packages.items():
        try:
            __import__(package)
            print(f"✓ {description:30s} ({package})")
        except ImportError as e:
            print(f"✗ {description:30s} ({package}) - FAILED")
            failed.append(package)

    print("-" * 60)

    if failed:
        print(f"\n✗ {len(failed)} package(s) failed to import:")
        for pkg in failed:
            print(f"  - {pkg}")
        print("\nPlease run: pip install -r requirements.txt")
        return False
    else:
        print("\n✓ All packages imported successfully!")
        return True


def test_gpu():
    """Test GPU configuration."""
    print("\n\nTesting GPU configuration...")
    print("-" * 60)

    try:
        import torch_directml
        from gpu_config import GPUConfig

        # Test device 1 (discrete GPU)
        gpu = GPUConfig(device_id=1)
        device = gpu.setup_device()

        print("✓ GPU Device 1 initialized successfully")
        print(f"  Device: {device}")
        print(f"  Name: {gpu.device_name}")

        # Try device 0 as well
        print("\nTrying Device 0 (integrated GPU)...")
        try:
            gpu0 = GPUConfig(device_id=0)
            device0 = gpu0.setup_device()
            print(f"✓ GPU Device 0 also available: {device0}")
        except Exception as e:
            print(f"  Device 0 not available (this is okay)")

        print("-" * 60)
        print("\n✓ GPU configuration test passed!")
        return True

    except Exception as e:
        print(f"✗ GPU test failed: {e}")
        print("-" * 60)
        print("\nPossible solutions:")
        print("  1. Update your AMD GPU drivers")
        print("  2. Verify DirectML is installed: pip install torch-directml")
        print("  3. Check if your GPU supports DirectML")
        return False


def test_config():
    """Test configuration file."""
    print("\n\nTesting configuration...")
    print("-" * 60)

    try:
        import config

        print(f"✓ Configuration loaded successfully")
        print(f"  GPU Device ID: {config.GPU_DEVICE_ID}")
        print(f"  Model: {config.MODEL_ID}")
        print(f"  Server: {config.SERVER_HOST}:{config.SERVER_PORT}")
        print(f"  Output Directory: {config.OUTPUT_DIR}")

        print("-" * 60)
        print("\n✓ Configuration test passed!")
        return True

    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        print("-" * 60)
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("STABLE DIFFUSION INSTALLATION TEST")
    print("="*60 + "\n")

    results = []

    # Run tests
    results.append(("Package Imports", test_imports()))
    results.append(("GPU Configuration", test_gpu()))
    results.append(("Configuration File", test_config()))

    # Summary
    print("\n\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    all_passed = True
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:30s}: {status}")
        if not passed:
            all_passed = False

    print("="*60)

    if all_passed:
        print("\n✓ All tests passed! Your installation is ready.")
        print("\nNext steps:")
        print("  1. Run: python app.py")
        print("  2. Open: http://127.0.0.1:7860")
        print("  3. Start generating images!")
        return 0
    else:
        print("\n✗ Some tests failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("  - Run: pip install -r requirements.txt")
        print("  - Update GPU drivers")
        print("  - Check Python version (3.10 recommended)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
