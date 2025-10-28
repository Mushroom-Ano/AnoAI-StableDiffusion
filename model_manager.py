"""
Model Manager Module

Handles discovery and loading of local and remote Stable Diffusion models.
"""

import os
from pathlib import Path
from typing import List, Dict, Tuple
import json


class ModelManager:
    """Manages local and remote Stable Diffusion models."""

    def __init__(self, models_dir: str = "models"):
        """
        Initialize the model manager.

        Args:
            models_dir: Directory containing local models
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)

    def scan_local_models(self) -> List[Dict[str, str]]:
        """
        Scan the models directory for available models.

        Returns:
            List of dicts with 'name' and 'path' keys
        """
        models = []

        if not self.models_dir.exists():
            return models

        # Scan for model directories (diffusers format)
        for item in self.models_dir.iterdir():
            if item.is_dir():
                # Check if it's a valid diffusers model
                if self._is_valid_diffusers_model(item):
                    models.append({
                        'name': item.name,
                        'path': str(item.resolve()),  # Use absolute path
                        'type': 'diffusers',
                        'display_name': f"📁 {item.name} (Local)"
                    })

            # Check for .safetensors or .ckpt files
            elif item.suffix in ['.safetensors', '.ckpt']:
                models.append({
                    'name': item.stem,
                    'path': str(item.resolve()),  # Use absolute path
                    'type': 'checkpoint',
                    'display_name': f"💾 {item.stem} (Checkpoint)"
                })

        return models

    def _is_valid_diffusers_model(self, model_path: Path) -> bool:
        """
        Check if a directory contains a valid diffusers model.

        Args:
            model_path: Path to check

        Returns:
            True if valid diffusers model
        """
        # Check for model_index.json (required for diffusers)
        model_index = model_path / "model_index.json"
        return model_index.exists()

    def get_default_models(self) -> List[Dict[str, str]]:
        """
        Get a list of popular HuggingFace models.

        Returns:
            List of dicts with 'name' and 'path' keys
        """
        return [
            # SD 1.5 Models
            {
                'name': 'stable-diffusion-v1-5',
                'path': 'runwayml/stable-diffusion-v1-5',
                'type': 'huggingface',
                'display_name': '☁️ Stable Diffusion v1.5 (HuggingFace)'
            },
            {
                'name': 'stable-diffusion-2-1',
                'path': 'stabilityai/stable-diffusion-2-1',
                'type': 'huggingface',
                'display_name': '☁️ Stable Diffusion v2.1 (HuggingFace)'
            },
            {
                'name': 'openjourney-v4',
                'path': 'prompthero/openjourney',
                'type': 'huggingface',
                'display_name': '☁️ OpenJourney v4 (HuggingFace)'
            },
            {
                'name': 'analog-diffusion',
                'path': 'wavymulder/Analog-Diffusion',
                'type': 'huggingface',
                'display_name': '☁️ Analog Diffusion (HuggingFace)'
            },
            {
                'name': 'dreamlike-photoreal',
                'path': 'dreamlike-art/dreamlike-photoreal-2.0',
                'type': 'huggingface',
                'display_name': '☁️ Dreamlike Photoreal 2.0 (HuggingFace)'
            },
            # SDXL Models
            {
                'name': 'stable-diffusion-xl-base',
                'path': 'stabilityai/stable-diffusion-xl-base-1.0',
                'type': 'huggingface',
                'display_name': '☁️ SDXL Base 1.0 (HuggingFace)'
            },
        ]

    def get_all_models(self) -> List[Dict[str, str]]:
        """
        Get all available models (local + HuggingFace).

        Returns:
            List of all available models
        """
        local_models = self.scan_local_models()
        default_models = self.get_default_models()

        # Combine, with local models first
        all_models = local_models + default_models

        return all_models

    def get_model_choices(self) -> Tuple[List[str], Dict[str, str]]:
        """
        Get model choices for UI dropdown.

        Returns:
            Tuple of (display_names, name_to_path_mapping)
        """
        models = self.get_all_models()

        display_names = [m['display_name'] for m in models]
        name_to_path = {m['display_name']: m['path'] for m in models}

        return display_names, name_to_path

    def get_model_info(self, model_path: str) -> Dict[str, str]:
        """
        Get information about a model.

        Args:
            model_path: Path or HuggingFace ID

        Returns:
            Dict with model information
        """
        path_obj = Path(model_path)

        info = {
            'path': model_path,
            'is_local': path_obj.exists(),
            'type': 'unknown'
        }

        if info['is_local']:
            if path_obj.is_dir():
                info['type'] = 'diffusers'
            elif path_obj.suffix in ['.safetensors', '.ckpt']:
                info['type'] = 'checkpoint'
        else:
            # Assume it's a HuggingFace model ID
            info['type'] = 'huggingface'

        return info


# Convenience function
def get_model_manager(models_dir: str = "models") -> ModelManager:
    """
    Get a ModelManager instance.

    Args:
        models_dir: Directory containing models

    Returns:
        ModelManager instance
    """
    return ModelManager(models_dir)


if __name__ == "__main__":
    # Test the model manager
    manager = ModelManager()

    print("Scanning for models...")
    print("\nLocal Models:")
    for model in manager.scan_local_models():
        print(f"  - {model['display_name']}")
        print(f"    Path: {model['path']}")

    print("\nAll Available Models:")
    for model in manager.get_all_models():
        print(f"  - {model['display_name']}")
