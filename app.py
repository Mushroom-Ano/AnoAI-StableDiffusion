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
        self.device = None
        self.generator = None
        self.model_loaded = False

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
            Generate high-quality images from text descriptions using AI.
            """
        )

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

        # Connect buttons to functions
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
