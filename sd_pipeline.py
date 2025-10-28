"""
Stable Diffusion Pipeline Wrapper

This module wraps the Hugging Face diffusers pipeline with DirectML support,
providing a clean interface for image generation with error handling.
"""

import os
from typing import List, Optional, Callable
from pathlib import Path
import random

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import torch_directml


class StableDiffusionGenerator:
    """Wrapper class for Stable Diffusion image generation with DirectML."""

    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        device=None,
        use_safetensors: bool = True
    ):
        """
        Initialize the Stable Diffusion pipeline.

        Args:
            model_id: Hugging Face model ID or local path
            device: DirectML device object (from gpu_config)
            use_safetensors: Whether to use safetensors format (faster, safer)
        """
        self.model_id = model_id
        self.device = device
        self.pipeline = None
        self.use_safetensors = use_safetensors

    def load_model(self, progress_callback: Optional[Callable] = None):
        """
        Load the Stable Diffusion model.

        Args:
            progress_callback: Optional callback function for progress updates
        """
        try:
            if progress_callback:
                progress_callback("Loading model from Hugging Face...")

            # Load the pipeline with DirectML device
            # torch_dtype is not specified as DirectML handles precision automatically
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                use_safetensors=self.use_safetensors,
            )

            if progress_callback:
                progress_callback("Moving model to GPU...")

            # Move pipeline to DirectML device
            # This transfers all model weights to the AMD GPU
            if self.device is not None:
                self.pipeline = self.pipeline.to(self.device)

            # Use DPM++ solver for better quality and faster generation
            # This scheduler typically produces good results in 20-25 steps
            self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipeline.scheduler.config
            )

            # Enable memory-efficient attention if available
            # This reduces VRAM usage for large images
            try:
                self.pipeline.enable_attention_slicing()
            except:
                pass  # Not critical if this fails

            if progress_callback:
                progress_callback("Model loaded successfully!")

            print(f"Model '{self.model_id}' loaded successfully on DirectML device")

        except Exception as e:
            error_msg = f"Failed to load model: {str(e)}"
            print(error_msg)
            if progress_callback:
                progress_callback(error_msg)
            raise

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        width: int = 512,
        height: int = 512,
        seed: Optional[int] = None,
        num_images: int = 1,
        progress_callback: Optional[Callable] = None
    ) -> List[Image.Image]:
        """
        Generate images from text prompt.

        Args:
            prompt: Text description of desired image
            negative_prompt: Things to avoid in the image
            num_inference_steps: Number of denoising steps (20-50 recommended)
            guidance_scale: How closely to follow the prompt (7-12 recommended)
            width: Image width in pixels (must be divisible by 8)
            height: Image height in pixels (must be divisible by 8)
            seed: Random seed for reproducibility (None = random)
            num_images: Number of images to generate
            progress_callback: Optional callback for progress updates

        Returns:
            List of generated PIL Images
        """
        if self.pipeline is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Validate dimensions (must be divisible by 8 for Stable Diffusion)
        if width % 8 != 0 or height % 8 != 0:
            raise ValueError("Width and height must be divisible by 8")

        # Handle seed
        if seed is None or seed == -1:
            seed = random.randint(0, 2**32 - 1)

        # Create generator for reproducibility
        # Note: DirectML may not support generator fully, but we try anyway
        generator = None
        try:
            generator = torch_directml.Generator(self.device).manual_seed(seed)
        except:
            # If generator fails, we'll still proceed without it
            print(f"Note: Using seed {seed} but generator may not be fully supported on DirectML")

        try:
            if progress_callback:
                progress_callback(f"Generating {num_images} image(s) with seed {seed}...")

            print(f"  > Preparing generation parameters...")
            print(f"     Prompt: {prompt[:50]}...")
            print(f"     Steps: {num_inference_steps}, Guidance: {guidance_scale}")
            print(f"     Size: {width}x{height}")

            # Prepare generation parameters
            gen_kwargs = {
                "prompt": prompt,
                "negative_prompt": negative_prompt if negative_prompt else None,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "width": width,
                "height": height,
                "num_images_per_prompt": num_images,
            }

            # Only add generator if it was successfully created
            # DirectML may not fully support generators
            if generator is not None:
                gen_kwargs["generator"] = generator
                print(f"  > Generator attached: {generator}")
            else:
                print(f"  > No generator (using random seed)")

            print(f"  > Calling Stable Diffusion pipeline...")
            print(f"     This will take 30-60 seconds on first run...")

            # Generate images
            output = self.pipeline(**gen_kwargs)

            print(f"  > Pipeline completed! Processing output...")
            images = output.images
            print(f"  > Got {len(images)} image(s)")

            if progress_callback:
                progress_callback(f"Successfully generated {len(images)} image(s)!")

            return images

        except Exception as e:
            error_msg = f"Generation failed: {str(e)}"
            print(error_msg)
            if progress_callback:
                progress_callback(error_msg)
            raise

    def save_images(
        self,
        images: List[Image.Image],
        output_dir: str = "outputs",
        prefix: str = "sd_image"
    ) -> List[str]:
        """
        Save generated images to disk.

        Args:
            images: List of PIL Images to save
            output_dir: Directory to save images in
            prefix: Filename prefix

        Returns:
            List of saved file paths
        """
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved_paths = []

        for i, image in enumerate(images):
            # Generate unique filename with timestamp
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{prefix}_{timestamp}_{i+1}.png"
            filepath = output_path / filename

            # Save image
            image.save(filepath)
            saved_paths.append(str(filepath))

            print(f"Saved: {filepath}")

        return saved_paths

    def unload_model(self):
        """Unload the model to free up GPU memory."""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            print("Model unloaded from GPU")


# Convenience function for quick testing
def quick_generate(
    prompt: str,
    device,
    model_id: str = "runwayml/stable-diffusion-v1-5",
    **kwargs
) -> List[Image.Image]:
    """
    Quick generation function for testing.

    Args:
        prompt: Text prompt
        device: DirectML device
        model_id: Model to use
        **kwargs: Additional generation parameters

    Returns:
        List of generated images
    """
    generator = StableDiffusionGenerator(model_id=model_id, device=device)
    generator.load_model()
    images = generator.generate(prompt=prompt, **kwargs)
    return images
