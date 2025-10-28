"""
Stable Diffusion Pipeline Wrapper

This module wraps the Hugging Face diffusers pipeline with DirectML support,
providing a clean interface for image generation with error handling.
"""

import os
from typing import List, Optional, Callable
from pathlib import Path
import random

from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    DPMSolverMultistepScheduler
)
from PIL import Image
import torch_directml
import config
from gpu_config import print_memory_usage


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
        self.img2img_pipeline = None
        self.use_safetensors = use_safetensors
        self.is_sdxl = False  # Track if model is SDXL

    def load_model(self, progress_callback: Optional[Callable] = None):
        """
        Load the Stable Diffusion model.

        Args:
            progress_callback: Optional callback function for progress updates
        """
        try:
            # Detect if model is a checkpoint file (.safetensors or .ckpt)
            model_path = Path(self.model_id)
            is_checkpoint = model_path.suffix in ['.safetensors', '.ckpt']

            # Try to detect if model is SDXL-based
            # SDXL models are typically 6+ GB, SD 1.5 models are ~4 GB
            # Also check filename for common SDXL indicators
            if is_checkpoint and model_path.exists():
                file_size_gb = model_path.stat().st_size / (1024**3)
                model_name_lower = model_path.name.lower()

                # Heuristics for SDXL detection
                self.is_sdxl = (
                    file_size_gb > 5.5 or  # SDXL files are typically 6-7 GB
                    'sdxl' in model_name_lower or
                    'xl' in model_name_lower or
                    'pony' in model_name_lower  # Pony models are SDXL-based
                )

                if self.is_sdxl:
                    print(f"  > Detected SDXL-based model (size: {file_size_gb:.1f} GB)")
            elif not is_checkpoint:
                # For diffusers format, check if it contains SDXL in the path
                self.is_sdxl = 'xl' in self.model_id.lower() or 'sdxl' in self.model_id.lower()

            if progress_callback:
                if is_checkpoint:
                    model_type = "SDXL" if self.is_sdxl else "SD 1.5"
                    progress_callback(f"Loading {model_type} checkpoint: {model_path.name}...")
                else:
                    progress_callback("Loading model from Hugging Face...")

            # Load the pipeline with DirectML device
            # torch_dtype is not specified as DirectML handles precision automatically
            # Safety checker can be disabled via config to prevent false positives
            # Note: SDXL models don't support safety_checker
            pipeline_kwargs = {}

            if config.DISABLE_SAFETY_CHECKER and not self.is_sdxl:
                pipeline_kwargs["safety_checker"] = None
                pipeline_kwargs["requires_safety_checker"] = False

            # Choose the correct pipeline class based on model type
            if self.is_sdxl:
                pipeline_class = StableDiffusionXLPipeline
                img2img_pipeline_class = StableDiffusionXLImg2ImgPipeline
                print(f"  > Using SDXL pipeline")
            else:
                pipeline_class = StableDiffusionPipeline
                img2img_pipeline_class = StableDiffusionImg2ImgPipeline
                print(f"  > Using SD 1.5 pipeline")

            # Load based on model type
            if is_checkpoint:
                # Use from_single_file for checkpoint files
                print(f"  > Loading checkpoint file: {self.model_id}")
                try:
                    self.pipeline = pipeline_class.from_single_file(
                        self.model_id,
                        **pipeline_kwargs
                    )
                except AttributeError:
                    error_msg = (
                        "Your version of diffusers doesn't support loading checkpoint files directly. "
                        "Please convert your .safetensors/.ckpt file to diffusers format, or upgrade diffusers: "
                        "pip install --upgrade diffusers"
                    )
                    print(f"[ERROR] {error_msg}")
                    if progress_callback:
                        progress_callback(f"[ERROR] {error_msg}")
                    raise ValueError(error_msg)
            else:
                # Use from_pretrained for diffusers format or HuggingFace models
                pipeline_kwargs["use_safetensors"] = self.use_safetensors
                print(f"  > Loading diffusers model: {self.model_id}")
                self.pipeline = pipeline_class.from_pretrained(
                    self.model_id,
                    **pipeline_kwargs
                )

            # Enable memory-efficient attention if available
            # This reduces VRAM usage for large images
            if self.is_sdxl and hasattr(config, 'SDXL_ENABLE_ATTENTION_SLICING'):
                # For SDXL, use max attention slicing for maximum memory savings
                try:
                    if config.SDXL_ENABLE_ATTENTION_SLICING == "max":
                        self.pipeline.enable_attention_slicing(slice_size="max")
                        print(f"  > Attention slicing enabled (max memory savings)")
                    else:
                        self.pipeline.enable_attention_slicing()
                        print(f"  > Attention slicing enabled")
                except Exception as e:
                    print(f"  > Attention slicing failed: {e}")
            else:
                try:
                    self.pipeline.enable_attention_slicing()
                except:
                    pass  # Not critical if this fails

            # For SDXL models, enable additional memory optimizations
            use_cpu_offload = self.is_sdxl and config.SDXL_ENABLE_MODEL_CPU_OFFLOAD
            use_sequential_offload = (
                self.is_sdxl and
                hasattr(config, 'SDXL_ENABLE_SEQUENTIAL_CPU_OFFLOAD') and
                config.SDXL_ENABLE_SEQUENTIAL_CPU_OFFLOAD
            )

            if self.is_sdxl:
                print(f"  > Enabling SDXL memory optimizations (DirectML needs aggressive settings)...")

                # Enable VAE slicing to reduce memory usage
                if config.SDXL_ENABLE_VAE_SLICING:
                    try:
                        self.pipeline.enable_vae_slicing()
                        print(f"    - VAE slicing enabled")
                    except Exception as e:
                        print(f"    - VAE slicing failed: {e}")

                # Enable VAE tiling for very large images
                if config.SDXL_ENABLE_VAE_TILING:
                    try:
                        self.pipeline.enable_vae_tiling()
                        print(f"    - VAE tiling enabled")
                    except Exception as e:
                        print(f"    - VAE tiling failed: {e}")

            if progress_callback:
                progress_callback("Moving model to GPU...")

            # Move pipeline to DirectML device OR use CPU offload (not both)
            # CPU offload manages device placement itself
            if use_sequential_offload:
                # Sequential CPU offload is most aggressive - loads components one at a time
                # Saves maximum VRAM but slower (good for DirectML with SDXL)
                try:
                    self.pipeline.enable_sequential_cpu_offload(gpu_id=config.GPU_DEVICE_ID)
                    print(f"    - Sequential CPU offload enabled (minimal VRAM, slower)")
                    print(f"    - This trades speed for memory - expect 2-3x slower generation")
                except Exception as e:
                    print(f"    - Sequential offload failed: {e}")
                    # Fall back to regular offload or GPU placement
                    if use_cpu_offload:
                        try:
                            self.pipeline.enable_model_cpu_offload(gpu_id=config.GPU_DEVICE_ID)
                            print(f"    - Model CPU offload enabled (fallback)")
                        except Exception as e2:
                            print(f"    - CPU offload failed: {e2}, using direct GPU placement")
                            if self.device is not None:
                                self.pipeline = self.pipeline.to(self.device)
                    else:
                        if self.device is not None:
                            self.pipeline = self.pipeline.to(self.device)
            elif use_cpu_offload:
                # Regular model CPU offload - moderate VRAM savings
                try:
                    self.pipeline.enable_model_cpu_offload(gpu_id=config.GPU_DEVICE_ID)
                    print(f"    - Model CPU offload enabled")
                except Exception as e:
                    print(f"    - CPU offload failed: {e}, using direct GPU placement")
                    if self.device is not None:
                        self.pipeline = self.pipeline.to(self.device)
            else:
                # Move all model components to DirectML device
                if self.device is not None:
                    self.pipeline = self.pipeline.to(self.device)
                    print(f"  > Model moved to GPU (Device {config.GPU_DEVICE_ID})")

            # Use DPM++ solver for better quality and faster generation
            # This scheduler typically produces good results in 20-25 steps
            self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipeline.scheduler.config
            )

            if progress_callback:
                progress_callback("Loading image-to-image pipeline...")

            # Load img2img pipeline using the same components
            # This reuses the model weights, so it's memory efficient
            img2img_kwargs = {
                "vae": self.pipeline.vae,
                "text_encoder": self.pipeline.text_encoder,
                "tokenizer": self.pipeline.tokenizer,
                "unet": self.pipeline.unet,
                "scheduler": self.pipeline.scheduler,
            }

            # Add feature_extractor only for non-SDXL models
            if not self.is_sdxl:
                img2img_kwargs["feature_extractor"] = self.pipeline.feature_extractor

            # SDXL models have additional components
            if self.is_sdxl:
                img2img_kwargs["text_encoder_2"] = self.pipeline.text_encoder_2
                img2img_kwargs["tokenizer_2"] = self.pipeline.tokenizer_2

            # Safety checker is only supported on SD 1.5 models, not SDXL
            if not self.is_sdxl:
                if config.DISABLE_SAFETY_CHECKER:
                    img2img_kwargs["safety_checker"] = None
                    img2img_kwargs["requires_safety_checker"] = False
                else:
                    img2img_kwargs["safety_checker"] = self.pipeline.safety_checker

            self.img2img_pipeline = img2img_pipeline_class(**img2img_kwargs)

            # Enable attention slicing for img2img too
            if self.is_sdxl and hasattr(config, 'SDXL_ENABLE_ATTENTION_SLICING'):
                try:
                    if config.SDXL_ENABLE_ATTENTION_SLICING == "max":
                        self.img2img_pipeline.enable_attention_slicing(slice_size="max")
                    else:
                        self.img2img_pipeline.enable_attention_slicing()
                except:
                    pass
            else:
                try:
                    self.img2img_pipeline.enable_attention_slicing()
                except:
                    pass

            # Apply SDXL memory optimizations to img2img pipeline as well
            if self.is_sdxl:
                if config.SDXL_ENABLE_VAE_SLICING:
                    try:
                        self.img2img_pipeline.enable_vae_slicing()
                    except:
                        pass
                if config.SDXL_ENABLE_VAE_TILING:
                    try:
                        self.img2img_pipeline.enable_vae_tiling()
                    except:
                        pass
                # CPU offload for img2img is handled by sharing components from main pipeline
                # No need to call enable_model_cpu_offload/sequential again as they share components

            if progress_callback:
                progress_callback("Model loaded successfully!")

            print(f"\nModel '{self.model_id}' loaded successfully on DirectML device")
            print(f"Both text-to-image and image-to-image pipelines ready")

            # Show memory usage after model loading
            print_memory_usage("After Model Load")
            print(f"\n[IMPORTANT] If System RAM usage is high, model may be in RAM instead of GPU VRAM!")
            print(f"            Model should be using GPU VRAM (16GB), not system RAM.")

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

        # Show which model is being used
        model_type = "SDXL" if self.is_sdxl else "SD 1.5"
        print(f"\n{'='*70}")
        print(f"GENERATING WITH: {self.model_id}")
        print(f"Model Type: {model_type}")
        print(f"{'='*70}")

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

    def generate_img2img(
        self,
        init_image: Image.Image,
        prompt: str,
        negative_prompt: str = "",
        strength: float = 0.75,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        num_images: int = 1,
        max_dimension: int = 768,
        progress_callback: Optional[Callable] = None
    ) -> List[Image.Image]:
        """
        Generate images from an initial image and text prompt (image-to-image).

        Args:
            init_image: Starting image (PIL Image)
            prompt: Text description of desired modifications
            negative_prompt: Things to avoid in the image
            strength: How much to transform the image (0.0=no change, 1.0=completely new)
                     Recommended: 0.3-0.5 for subtle changes, 0.6-0.8 for major changes
            num_inference_steps: Number of denoising steps (recommended: 50 for img2img)
            guidance_scale: How closely to follow the prompt (7-12 recommended)
            seed: Random seed for reproducibility (None = random)
            num_images: Number of variations to generate
            max_dimension: Maximum width or height (resizes if larger to save VRAM)
            progress_callback: Optional callback for progress updates

        Returns:
            List of generated PIL Images
        """
        if self.img2img_pipeline is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Validate strength
        if strength < 0.0 or strength > 1.0:
            raise ValueError("Strength must be between 0.0 and 1.0")

        # Resize image if too large to prevent OOM errors
        original_size = init_image.size
        if max(init_image.size) > max_dimension:
            # Calculate new size maintaining aspect ratio
            width, height = init_image.size
            if width > height:
                new_width = max_dimension
                new_height = int((max_dimension / width) * height)
            else:
                new_height = max_dimension
                new_width = int((max_dimension / height) * width)

            # Ensure dimensions are divisible by 8
            new_width = (new_width // 8) * 8
            new_height = (new_height // 8) * 8

            print(f"  > Resizing image from {original_size} to ({new_width}, {new_height}) to save VRAM")
            init_image = init_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

            if progress_callback:
                progress_callback(f"Resized image to {new_width}x{new_height} to prevent memory issues")
        else:
            # Still ensure dimensions are divisible by 8
            width, height = init_image.size
            new_width = (width // 8) * 8
            new_height = (height // 8) * 8
            if (new_width, new_height) != (width, height):
                print(f"  > Adjusting dimensions from {init_image.size} to ({new_width}, {new_height})")
                init_image = init_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Handle seed
        if seed is None or seed == -1:
            seed = random.randint(0, 2**32 - 1)

        # Create generator for reproducibility
        generator = None
        try:
            generator = torch_directml.Generator(self.device).manual_seed(seed)
        except:
            print(f"Note: Using seed {seed} but generator may not be fully supported on DirectML")

        try:
            if progress_callback:
                progress_callback(f"Transforming image with strength {strength}...")

            print(f"  > Preparing img2img generation...")
            print(f"     Prompt: {prompt[:50]}...")
            print(f"     Strength: {strength}, Steps: {num_inference_steps}")
            print(f"     Input image size: {init_image.size}")

            # Prepare generation parameters
            gen_kwargs = {
                "prompt": prompt,
                "image": init_image,
                "negative_prompt": negative_prompt if negative_prompt else None,
                "strength": strength,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "num_images_per_prompt": num_images,
            }

            # Only add generator if it was successfully created
            if generator is not None:
                gen_kwargs["generator"] = generator
                print(f"  > Generator attached: {generator}")
            else:
                print(f"  > No generator (using random seed)")

            print(f"  > Calling img2img pipeline...")
            print(f"     This will take 30-90 seconds...")

            # Generate images
            output = self.img2img_pipeline(**gen_kwargs)

            print(f"  > Pipeline completed! Processing output...")
            images = output.images
            print(f"  > Got {len(images)} transformed image(s)")

            if progress_callback:
                progress_callback(f"Successfully generated {len(images)} image(s)!")

            return images

        except Exception as e:
            error_msg = f"Img2img generation failed: {str(e)}"
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
