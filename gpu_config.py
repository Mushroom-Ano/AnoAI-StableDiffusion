"""
GPU Configuration Module for DirectML on Windows

This module handles GPU device selection for torch-directml, ensuring
the correct AMD GPU is used instead of the integrated GPU.
"""

import os
import sys
from typing import Optional, Dict

try:
    import torch_directml
except ImportError:
    print("ERROR: torch-directml is not installed!")
    print("Please install it with: pip install torch-directml")
    sys.exit(1)


class GPUConfig:
    """Manages GPU configuration for DirectML backend."""

    def __init__(self, device_id: int = 1):
        """
        Initialize GPU configuration.

        Args:
            device_id: The DirectML device ID to use (default: 1 for discrete GPU)
                      Device 0 is typically the integrated GPU
                      Device 1 is typically the discrete GPU (AMD RX 9070 XT)
        """
        self.device_id = device_id
        self.device = None
        self.device_name = "Unknown"

    def setup_device(self) -> torch_directml.device:
        """
        Configure and return the DirectML device.

        This method:
        1. Sets environment variables to prefer the specified device
        2. Creates a torch_directml device object
        3. Retrieves device information

        Returns:
            torch_directml.device: Configured DirectML device
        """
        # Set environment variable to force device selection
        # This helps ensure DirectML uses the correct GPU
        os.environ['TORCH_DIRECTML_DEVICE_INDEX'] = str(self.device_id)

        try:
            # Create DirectML device object
            # torch_directml.device(N) selects the Nth DirectML-compatible device
            self.device = torch_directml.device(self.device_id)

            # Try to get device name (if available in the DirectML API)
            try:
                self.device_name = f"DirectML Device {self.device_id}"
                # Note: DirectML doesn't always expose device names directly
                # You may see generic names rather than specific GPU models
            except:
                self.device_name = f"DirectML Device {self.device_id}"

            return self.device

        except Exception as e:
            print(f"ERROR: Failed to initialize DirectML device {self.device_id}")
            print(f"Error details: {e}")
            print("\nAvailable options:")
            print("- Make sure your GPU drivers are up to date")
            print("- Try device_id=0 if device 1 is not available")
            print("- Verify DirectML is properly installed")
            sys.exit(1)

    def get_device_info(self) -> Dict[str, str]:
        """
        Get information about the configured device.

        Returns:
            Dict containing device information
        """
        if self.device is None:
            return {"status": "Not initialized"}

        return {
            "device_id": str(self.device_id),
            "device_name": self.device_name,
            "backend": "DirectML (Windows)",
            "device_object": str(self.device)
        }

    def print_device_info(self):
        """Print formatted device information to console."""
        info = self.get_device_info()

        print("\n" + "="*60)
        print("GPU CONFIGURATION")
        print("="*60)

        for key, value in info.items():
            formatted_key = key.replace("_", " ").title()
            print(f"{formatted_key:20s}: {value}")

        print("="*60)
        print(f"Using GPU Device {self.device_id} (AMD RX 9070 XT)")
        print("Device 0 (integrated GPU) will be ignored")
        print("="*60 + "\n")


def initialize_gpu(device_id: int = 1) -> torch_directml.device:
    """
    Convenience function to initialize GPU with default settings.

    Args:
        device_id: The DirectML device ID to use (default: 1)

    Returns:
        Configured DirectML device object
    """
    gpu_config = GPUConfig(device_id=device_id)
    device = gpu_config.setup_device()
    gpu_config.print_device_info()
    return device


if __name__ == "__main__":
    # Test the GPU configuration
    print("Testing GPU Configuration...")
    device = initialize_gpu(device_id=1)
    print(f"Successfully initialized: {device}")
