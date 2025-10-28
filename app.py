"""
Stable Diffusion Web UI with DirectML Support

A clean, user-friendly web interface for generating images with Stable Diffusion
on AMD GPUs using DirectML on Windows.
"""

import gradio as gr
from typing import List, Optional, Tuple
from PIL import Image
import traceback
from datetime import datetime
from pathlib import Path

from gpu_config import initialize_gpu
from sd_pipeline import StableDiffusionGenerator
from model_manager import ModelManager
import config


class StableDiffusionApp:
    """Main application class for the Stable Diffusion web UI."""

    def __init__(self, device_id: int = 1, model_id: str = "runwayml/stable-diffusion-v1-5"):
        """
        Initialize the application.

        Args:
            device_id: DirectML device ID (1 = discrete GPU)
            model_id: Hugging Face model ID to use
        """
        self.device_id = device_id
        self.model_id = model_id
        self.current_model = model_id
        self.device = None
        self.generator = None
        self.model_loaded = False
        self.model_manager = ModelManager(config.MODELS_DIR)

    def initialize(self) -> str:
        """
        Initialize GPU and load model.

        Returns:
            Status message
        """
        try:
            # Initialize GPU
            print("Initializing GPU...")
            self.device = initialize_gpu(device_id=self.device_id)

            # Create generator
            print(f"Creating Stable Diffusion generator with model: {self.model_id}")
            self.generator = StableDiffusionGenerator(
                model_id=self.model_id,
                device=self.device
            )

            # Load model
            print("Loading Stable Diffusion model (this may take a few minutes)...")
            self.generator.load_model()

            self.model_loaded = True
            return "[OK] Model loaded successfully! Ready to generate images."

        except Exception as e:
            error_msg = f"[ERROR] Initialization failed: {str(e)}"
            print(error_msg)
            print(traceback.format_exc())
            return error_msg

    def switch_model(self, new_model_id: str) -> str:
        """
        Switch to a different Stable Diffusion model.

        Args:
            new_model_id: New model ID or path

        Returns:
            Status message
        """
        if new_model_id == self.current_model:
            return f"[INFO] Model '{new_model_id}' is already loaded."

        try:
            print(f"\n{'='*70}")
            print(f"SWITCHING MODEL")
            print(f"From: {self.current_model}")
            print(f"To: {new_model_id}")
            print(f"{'='*70}\n")

            # Unload current model
            if self.generator:
                print("Unloading current model...")
                self.generator.unload_model()
                self.generator = None
                self.model_loaded = False

            # Create new generator with new model
            print(f"Loading new model: {new_model_id}")
            self.generator = StableDiffusionGenerator(
                model_id=new_model_id,
                device=self.device
            )

            # Load model
            print("Loading model (this may take a minute)...")
            self.generator.load_model()

            self.current_model = new_model_id
            self.model_loaded = True

            return f"[OK] Successfully switched to model: {new_model_id}"

        except Exception as e:
            error_msg = f"[ERROR] Failed to switch model: {str(e)}"
            print(error_msg)
            print(traceback.format_exc())

            # Try to reload the previous model
            try:
                print(f"Attempting to reload previous model: {self.current_model}")
                self.generator = StableDiffusionGenerator(
                    model_id=self.current_model,
                    device=self.device
                )
                self.generator.load_model()
                self.model_loaded = True
                return f"{error_msg}\n[INFO] Restored previous model: {self.current_model}"
            except:
                self.model_loaded = False
                return f"{error_msg}\n[ERROR] Could not restore previous model. Please restart the application."

    def generate_images(
        self,
        prompt: str,
        negative_prompt: str,
        steps: int,
        guidance_scale: float,
        width: int,
        height: int,
        seed: int,
        num_images: int,
        progress: Optional[gr.Progress] = None
    ) -> Tuple[List[Image.Image], str]:
        """
        Generate images based on user inputs.

        Args:
            prompt: Text description of desired image
            negative_prompt: Things to avoid
            steps: Number of denoising steps
            guidance_scale: How closely to follow prompt
            width: Image width
            height: Image height
            seed: Random seed (-1 for random)
            num_images: Number of images to generate
            progress: Gradio progress tracker

        Returns:
            Tuple of (list of images, status message)
        """
        if not self.model_loaded:
            return [], "[ERROR] Model not loaded. Please restart the application."

        # Show current model info
        print(f"\n[INFO] Current model: {self.current_model}")
        if self.generator:
            model_type = "SDXL" if self.generator.is_sdxl else "SD 1.5"
            print(f"[INFO] Model type: {model_type}")

        # Validate inputs
        if not prompt or prompt.strip() == "":
            return [], "[ERROR] Please enter a prompt."

        if steps < 1 or steps > 150:
            return [], "[ERROR] Steps must be between 1 and 150."

        if guidance_scale < 1 or guidance_scale > 30:
            return [], "[ERROR] Guidance scale must be between 1 and 30."

        if num_images < 1 or num_images > 10:
            return [], "[ERROR] Number of images must be between 1 and 10."

        # Ensure dimensions are divisible by 8
        width = (width // 8) * 8
        height = (height // 8) * 8

        try:
            print("> Starting image generation process...")

            # Progress callback
            def progress_callback(msg):
                print(f"  Progress: {msg}")
                if progress is not None:
                    progress(0.5, desc=msg)

            if progress is not None:
                progress(0, desc="Starting generation...")

            print("> Calling pipeline.generate()...")
            # Generate images
            images = self.generator.generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                seed=seed,
                num_images=num_images,
                progress_callback=progress_callback
            )

            print(f"> Pipeline returned {len(images)} images")

            if progress is not None:
                progress(1.0, desc="Complete!")

            # Create status message
            actual_seed = seed if seed != -1 else "random"
            status = f"[OK] Generated {len(images)} image(s) successfully!\n"
            status += f"Seed: {actual_seed} | Steps: {steps} | Guidance: {guidance_scale}"

            return images, status

        except Exception as e:
            error_msg = f"[ERROR] Generation failed: {str(e)}"
            print(error_msg)
            print(traceback.format_exc())
            return [], error_msg

    def generate_img2img_images(
        self,
        init_image: Image.Image,
        prompt: str,
        negative_prompt: str,
        strength: float,
        max_dimension: int,
        steps: int,
        guidance_scale: float,
        seed: int,
        num_images: int,
        progress: Optional[gr.Progress] = None
    ) -> Tuple[List[Image.Image], str]:
        """
        Generate images from an input image (img2img).

        Args:
            init_image: Starting image
            prompt: Text description of modifications
            negative_prompt: Things to avoid
            strength: Transformation strength (0.0-1.0)
            steps: Number of denoising steps
            guidance_scale: How closely to follow prompt
            seed: Random seed (-1 for random)
            num_images: Number of images to generate
            progress: Gradio progress tracker

        Returns:
            Tuple of (list of images, status message)
        """
        if not self.model_loaded:
            return [], "[ERROR] Model not loaded. Please restart the application."

        # Validate inputs
        if init_image is None:
            return [], "[ERROR] Please upload an image."

        if not prompt or prompt.strip() == "":
            return [], "[ERROR] Please enter a prompt."

        if steps < 1 or steps > 150:
            return [], "[ERROR] Steps must be between 1 and 150."

        if guidance_scale < 1 or guidance_scale > 30:
            return [], "[ERROR] Guidance scale must be between 1 and 30."

        if strength < 0.0 or strength > 1.0:
            return [], "[ERROR] Strength must be between 0.0 and 1.0."

        if num_images < 1 or num_images > 10:
            return [], "[ERROR] Number of images must be between 1 and 10."

        try:
            print("> Starting img2img generation process...")

            # Progress callback
            def progress_callback(msg):
                print(f"  Progress: {msg}")
                if progress is not None:
                    progress(0.5, desc=msg)

            if progress is not None:
                progress(0, desc="Starting transformation...")

            print("> Calling img2img pipeline...")
            # Generate images
            images = self.generator.generate_img2img(
                init_image=init_image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                strength=strength,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                seed=seed,
                num_images=num_images,
                max_dimension=max_dimension,
                progress_callback=progress_callback
            )

            print(f"> Pipeline returned {len(images)} images")

            if progress is not None:
                progress(1.0, desc="Complete!")

            # Create status message
            actual_seed = seed if seed != -1 else "random"
            status = f"[OK] Transformed {len(images)} image(s) successfully!\n"
            status += f"Strength: {strength} | Seed: {actual_seed} | Steps: {steps} | Guidance: {guidance_scale}"

            return images, status

        except Exception as e:
            error_msg = f"[ERROR] Img2img generation failed: {str(e)}"
            print(error_msg)
            print(traceback.format_exc())
            return [], error_msg

    def save_images(
        self,
        images: Optional[List[Image.Image]],
        prompt: str
    ) -> str:
        """
        Save generated images to disk.

        Args:
            images: List of images to save
            prompt: Original prompt (used for filename)

        Returns:
            Status message with saved paths
        """
        if not images or len(images) == 0:
            return "[ERROR] No images to save."

        try:
            # Create safe filename from prompt
            safe_prompt = "".join(c for c in prompt[:50] if c.isalnum() or c in (' ', '_')).strip()
            safe_prompt = safe_prompt.replace(' ', '_')

            # Save images
            saved_paths = self.generator.save_images(
                images=images,
                output_dir=config.OUTPUT_DIR,
                prefix=safe_prompt if safe_prompt else "sd_image"
            )

            # Create status message
            status = f"[OK] Saved {len(saved_paths)} image(s):\n"
            for path in saved_paths:
                status += f"  - {path}\n"

            return status

        except Exception as e:
            error_msg = f"[ERROR] Failed to save images: {str(e)}"
            print(error_msg)
            return error_msg


def create_ui(app: StableDiffusionApp) -> gr.Blocks:
    """
    Create the Gradio web interface.

    Args:
        app: StableDiffusionApp instance

    Returns:
        Gradio Blocks interface
    """
    with gr.Blocks(
        title="Stable Diffusion with DirectML",
        theme=gr.themes.Soft()
    ) as demo:
        gr.Markdown(
            """
            # 🎨 Stable Diffusion Image Generator
            ### Powered by DirectML on AMD RX 9070 XT
            Generate high-quality images from text descriptions or transform existing images using AI.
            """
        )

        # Model Selector Section
        if config.ENABLE_MODEL_SWITCHING:
            with gr.Accordion("🔧 Model Settings", open=False):
                with gr.Row():
                    # Get available models
                    model_choices, model_mapping = app.model_manager.get_model_choices()

                    model_dropdown = gr.Dropdown(
                        choices=model_choices,
                        value=model_choices[0] if model_choices else None,
                        label="Select Model",
                        info="Choose from local models or download from HuggingFace"
                    )

                    load_model_btn = gr.Button(
                        "📥 Load Model",
                        variant="primary"
                    )

                model_status = gr.Textbox(
                    label="Model Status",
                    value=f"Current: {app.current_model}",
                    interactive=False,
                    lines=2
                )

        with gr.Tabs():
            # TEXT-TO-IMAGE TAB
            with gr.Tab("📝 Text-to-Image"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 📝 Prompts")

                        prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="A beautiful landscape with mountains and a lake at sunset...",
                            lines=3,
                            value=config.DEFAULT_PROMPT
                        )

                        negative_prompt = gr.Textbox(
                            label="Negative Prompt",
                            placeholder="Things to avoid (ugly, blurry, low quality...)",
                            lines=2,
                            value=config.DEFAULT_NEGATIVE_PROMPT
                        )

                        gr.Markdown("### ⚙️ Settings")

                        with gr.Row():
                            steps = gr.Slider(
                        minimum=1,
                        maximum=config.MAX_STEPS,
                        value=config.DEFAULT_STEPS,
                        step=1,
                        label="Steps",
                        info="More steps = better quality but slower"
                    )

                    guidance_scale = gr.Slider(
                        minimum=1,
                        maximum=config.MAX_GUIDANCE_SCALE,
                        value=config.DEFAULT_GUIDANCE_SCALE,
                        step=0.5,
                        label="Guidance Scale",
                        info="How closely to follow the prompt"
                    )

                with gr.Row():
                    width = gr.Slider(
                        minimum=256,
                        maximum=config.MAX_DIMENSION,
                        value=config.DEFAULT_WIDTH,
                        step=64,
                        label="Width",
                        info="Image width (divisible by 8)"
                    )

                    height = gr.Slider(
                        minimum=256,
                        maximum=config.MAX_DIMENSION,
                        value=config.DEFAULT_HEIGHT,
                        step=64,
                        label="Height",
                        info="Image height (divisible by 8)"
                    )

                with gr.Row():
                    seed = gr.Number(
                        label="Seed",
                        value=-1,
                        precision=0,
                        info="Use -1 for random seed"
                    )

                    num_images = gr.Slider(
                        minimum=1,
                        maximum=config.MAX_BATCH_SIZE,
                        value=config.DEFAULT_NUM_IMAGES,
                        step=1,
                        label="Number of Images",
                        info="Batch generation"
                    )

                generate_btn = gr.Button(
                    "🎨 Generate Images",
                    variant="primary"
                )

            with gr.Column(scale=1):
                gr.Markdown("### 🖼️ Generated Images")

                gallery = gr.Gallery(
                    label="Results",
                    show_label=False,
                    elem_id="gallery"
                )

                status = gr.Textbox(
                    label="Status",
                    lines=3,
                    interactive=False,
                    show_label=True
                )

                with gr.Row():
                    save_btn = gr.Button(
                        "💾 Save Images",
                        variant="secondary"
                    )

                    clear_btn = gr.Button(
                        "🗑️ Clear",
                        variant="secondary"
                    )

            # IMAGE-TO-IMAGE TAB
            with gr.Tab("🖼️ Image-to-Image"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 🖼️ Input Image")

                        img2img_image = gr.Image(
                            label="Upload Image",
                            type="pil",
                            sources=["upload", "clipboard"],
                            image_mode="RGB"
                        )

                        gr.Markdown("### 📝 Prompts")

                        img2img_prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="Transform this image into...",
                            lines=3,
                            value="Make it look like a watercolor painting, soft colors, artistic"
                        )

                        img2img_negative_prompt = gr.Textbox(
                            label="Negative Prompt",
                            placeholder="Things to avoid...",
                            lines=2,
                            value="ugly, blurry, low quality, distorted, deformed"
                        )

                        gr.Markdown("### ⚙️ Settings")

                        img2img_strength = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=config.IMG2IMG_DEFAULT_STRENGTH,
                            step=0.05,
                            label="Transformation Strength",
                            info="0=keep original, 1=completely new (0.3-0.8 recommended)"
                        )

                        img2img_max_size = gr.Slider(
                            minimum=512,
                            maximum=1024,
                            value=config.IMG2IMG_MAX_DIMENSION,
                            step=64,
                            label="Max Image Size",
                            info="Images larger than this will be resized (saves VRAM)"
                        )

                        with gr.Row():
                            img2img_steps = gr.Slider(
                                minimum=1,
                                maximum=config.MAX_STEPS,
                                value=50,
                                step=1,
                                label="Steps",
                                info="Recommended: 50 for img2img"
                            )

                            img2img_guidance = gr.Slider(
                                minimum=1,
                                maximum=config.MAX_GUIDANCE_SCALE,
                                value=7.5,
                                step=0.5,
                                label="Guidance Scale",
                                info="How closely to follow the prompt"
                            )

                        with gr.Row():
                            img2img_seed = gr.Number(
                                label="Seed",
                                value=-1,
                                precision=0,
                                info="Use -1 for random seed"
                            )

                            img2img_num_images = gr.Slider(
                                minimum=1,
                                maximum=config.MAX_BATCH_SIZE,
                                value=1,
                                step=1,
                                label="Number of Variations",
                                info="Generate multiple versions"
                            )

                        img2img_generate_btn = gr.Button(
                            "🎨 Transform Image",
                            variant="primary"
                        )

                    with gr.Column(scale=1):
                        gr.Markdown("### 🖼️ Transformed Images")

                        img2img_gallery = gr.Gallery(
                            label="Results",
                            show_label=False,
                            elem_id="img2img_gallery"
                        )

                        img2img_status = gr.Textbox(
                            label="Status",
                            lines=3,
                            interactive=False,
                            show_label=True
                        )

                        with gr.Row():
                            img2img_save_btn = gr.Button(
                                "💾 Save Images",
                                variant="secondary"
                            )

                            img2img_clear_btn = gr.Button(
                                "🗑️ Clear",
                                variant="secondary"
                            )

        gr.Markdown(
            """
            ---
            ### 💡 Tips:
            - **Prompts**: Be descriptive! Include details about style, lighting, mood, etc.
            - **Steps**: 20-30 steps usually give good results. 50+ for highest quality.
            - **Guidance Scale**: 7-12 works well for most images. Higher = more literal interpretation.
            - **Seed**: Use the same seed with same settings to reproduce images.
            - **Batch Generation**: Generate multiple variations at once.

            ### 🎯 Example Prompts:
            - "A cyberpunk city at night, neon lights, raining, cinematic, highly detailed"
            - "A cute cat wearing a wizard hat, fantasy art, digital painting"
            - "A cozy library with fireplace, warm lighting, books everywhere, photorealistic"
            """
        )

        # Event handlers
        def generate_wrapper(
            prompt_text,
            neg_prompt_text,
            steps_val,
            guidance_val,
            width_val,
            height_val,
            seed_val,
            num_images_val
        ):
            """Wrapper to handle generation and track images."""
            try:

                # Convert types to ensure compatibility
                images, msg = app.generate_images(
                    prompt_text,
                    neg_prompt_text,
                    int(steps_val),
                    float(guidance_val),
                    int(width_val),
                    int(height_val),
                    int(seed_val),
                    int(num_images_val),
                    None  # No progress tracking for now
                )

                # Store images for saving later
                generate_wrapper.last_images = images
                return images, msg
            except Exception as e:
                error_msg = f"[ERROR] Error in generate_wrapper: {str(e)}"
                print(error_msg, flush=True)
                import traceback
                traceback.print_exc()
                return [], error_msg

        generate_wrapper.last_images = []

        def save_wrapper(prompt_text):
            """Wrapper to save the last generated images."""
            return app.save_images(generate_wrapper.last_images, prompt_text)

        def clear_wrapper():
            """Clear the gallery and status."""
            generate_wrapper.last_images = []
            return [], ""

        # Image-to-Image wrappers
        def img2img_wrapper(
            image,
            prompt_text,
            neg_prompt_text,
            strength_val,
            max_size_val,
            steps_val,
            guidance_val,
            seed_val,
            num_images_val
        ):
            """Wrapper to handle img2img generation and track images."""
            try:
                # Convert types to ensure compatibility
                images, msg = app.generate_img2img_images(
                    image,
                    prompt_text,
                    neg_prompt_text,
                    float(strength_val),
                    int(max_size_val),
                    int(steps_val),
                    float(guidance_val),
                    int(seed_val),
                    int(num_images_val),
                    None  # No progress tracking for now
                )

                # Store images for saving later
                img2img_wrapper.last_images = images
                return images, msg
            except Exception as e:
                error_msg = f"[ERROR] Error in img2img_wrapper: {str(e)}"
                print(error_msg, flush=True)
                import traceback
                traceback.print_exc()
                return [], error_msg

        img2img_wrapper.last_images = []

        def img2img_save_wrapper(prompt_text):
            """Wrapper to save the last img2img generated images."""
            return app.save_images(img2img_wrapper.last_images, prompt_text)

        def img2img_clear_wrapper():
            """Clear the img2img gallery and status."""
            img2img_wrapper.last_images = []
            return [], ""

        # Model switching wrapper
        def switch_model_wrapper(model_display_name):
            """Wrapper to handle model switching from UI."""
            if not model_display_name:
                return "[ERROR] Please select a model."

            # Map display name to actual path
            model_path = model_mapping.get(model_display_name)
            if not model_path:
                return f"[ERROR] Could not find path for model: {model_display_name}"

            # Switch to the selected model
            print(f"\n[UI] Switching to model: {model_display_name}")
            print(f"[UI] Model path: {model_path}")
            status_msg = app.switch_model(model_path)

            # Update status to show current model
            final_status = f"{status_msg}\n\nCurrent Model: {app.current_model}"
            if app.generator:
                model_type = "SDXL" if app.generator.is_sdxl else "SD 1.5"
                final_status += f"\nType: {model_type}"

            return final_status

        # Connect buttons to functions
        # Text-to-Image connections
        generate_btn.click(
            fn=generate_wrapper,
            inputs=[
                prompt,
                negative_prompt,
                steps,
                guidance_scale,
                width,
                height,
                seed,
                num_images
            ],
            outputs=[gallery, status]
        )

        save_btn.click(
            fn=save_wrapper,
            inputs=[prompt],
            outputs=[status]
        )

        clear_btn.click(
            fn=clear_wrapper,
            inputs=[],
            outputs=[gallery, status]
        )

        # Image-to-Image connections
        img2img_generate_btn.click(
            fn=img2img_wrapper,
            inputs=[
                img2img_image,
                img2img_prompt,
                img2img_negative_prompt,
                img2img_strength,
                img2img_max_size,
                img2img_steps,
                img2img_guidance,
                img2img_seed,
                img2img_num_images
            ],
            outputs=[img2img_gallery, img2img_status]
        )

        img2img_save_btn.click(
            fn=img2img_save_wrapper,
            inputs=[img2img_prompt],
            outputs=[img2img_status]
        )

        img2img_clear_btn.click(
            fn=img2img_clear_wrapper,
            inputs=[],
            outputs=[img2img_gallery, img2img_status]
        )

        # Model switching connections (if enabled)
        if config.ENABLE_MODEL_SWITCHING:
            load_model_btn.click(
                fn=switch_model_wrapper,
                inputs=[model_dropdown],
                outputs=[model_status]
            )

    return demo


def main():
    """Main entry point for the application."""
    print("\n" + "="*70)
    print("STABLE DIFFUSION WITH DIRECTML")
    print("="*70 + "\n")

    # Create app instance
    app = StableDiffusionApp(
        device_id=config.GPU_DEVICE_ID,  # Force GPU device 1 (AMD RX 9070 XT)
        model_id=config.MODEL_ID  # You can change this in config.py
    )

    # Initialize GPU and model
    print("Initializing application...")
    init_status = app.initialize()
    print(f"\n{init_status}\n")

    if not app.model_loaded:
        print("ERROR: Failed to initialize. Please check the error messages above.")
        return

    # Create and launch UI
    print("Starting web interface...")
    demo = create_ui(app)

    # Enable queuing for progress tracking
    demo.queue()

    # Launch with configuration
    demo.launch(
        server_name=config.SERVER_HOST,  # Localhost only (change in config.py)
        server_port=config.SERVER_PORT,
        share=config.SHARE_PUBLICLY,  # Set to True to create a public link
        inbrowser=config.AUTO_OPEN_BROWSER,  # Automatically open in browser
        show_error=config.SHOW_ERRORS
    )


if __name__ == "__main__":
    main()
