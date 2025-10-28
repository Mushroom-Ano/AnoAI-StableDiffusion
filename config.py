"""
Configuration file for Stable Diffusion DirectML Application

Edit these settings to customize the application without modifying the main code.
"""

# ============================================================================
# GPU CONFIGURATION
# ============================================================================

# DirectML device ID
# Device 0: Usually integrated GPU
# Device 1: Usually discrete GPU (AMD RX 9070 XT)
GPU_DEVICE_ID = 1

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Local models directory (place your models here)
MODELS_DIR = "models"

# Default model to load on startup
# Can be:
#   - HuggingFace model ID: "runwayml/stable-diffusion-v1-5"
#   - Local model path: "models/my-custom-model"
#   - Relative path: "./models/stable-diffusion-v1-5"
MODEL_ID = "runwayml/stable-diffusion-v1-5"

# Use safetensors format (faster and safer)
USE_SAFETENSORS = True

# Disable NSFW safety checker (reduces false positives)
# Set to True if you want to enable the safety filter
DISABLE_SAFETY_CHECKER = True

# Enable model switching in UI (allows changing models without restart)
ENABLE_MODEL_SWITCHING = True

# ============================================================================
# UI CONFIGURATION
# ============================================================================

# Web server settings
SERVER_HOST = "127.0.0.1"  # Localhost only. Use "0.0.0.0" to allow network access
SERVER_PORT = 7860

# Create public share link (via Gradio)
SHARE_PUBLICLY = False

# Automatically open browser on startup
AUTO_OPEN_BROWSER = True

# Theme (options: "default", "soft", "monochrome")
UI_THEME = "soft"

# ============================================================================
# GENERATION DEFAULTS
# ============================================================================

# Default generation settings (users can override in UI)
DEFAULT_STEPS = 30
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 512
DEFAULT_NUM_IMAGES = 1

# Default prompts
DEFAULT_PROMPT = "A serene mountain landscape with a crystal clear lake reflecting the golden sunset, detailed, 4k, photorealistic"
DEFAULT_NEGATIVE_PROMPT = "ugly, blurry, low quality, distorted, deformed"

# ============================================================================
# OUTPUT CONFIGURATION
# ============================================================================

# Directory for saved images
OUTPUT_DIR = "outputs"

# Image format (PNG recommended for quality)
IMAGE_FORMAT = "PNG"

# ============================================================================
# ADVANCED SETTINGS
# ============================================================================

# Enable memory optimizations
ENABLE_ATTENTION_SLICING = True
ENABLE_VAE_SLICING = False  # Enable if you run out of memory

# SDXL-specific settings
# DirectML is less memory-efficient than CUDA, so we need aggressive optimizations
SDXL_ENABLE_MODEL_CPU_OFFLOAD = True  # Enable to save VRAM (trades speed for memory)
SDXL_ENABLE_SEQUENTIAL_CPU_OFFLOAD = True  # Most aggressive memory saving
SDXL_ENABLE_VAE_SLICING = True  # Reduces VRAM for VAE operations
SDXL_ENABLE_VAE_TILING = True  # Enables processing large images in tiles
SDXL_ENABLE_ATTENTION_SLICING = "max"  # Max memory savings for attention
SDXL_DEFAULT_WIDTH = 768  # Lower than native 1024 to save VRAM
SDXL_DEFAULT_HEIGHT = 768  # Can try 1024 once optimizations are working

# Maximum values (safety limits)
MAX_STEPS = 150
MAX_GUIDANCE_SCALE = 30
MAX_DIMENSION = 1024
MAX_BATCH_SIZE = 10

# Image-to-Image settings
IMG2IMG_MAX_DIMENSION = 768  # Resize larger images to this to prevent OOM errors
IMG2IMG_DEFAULT_STRENGTH = 0.75

# ============================================================================
# DEBUGGING
# ============================================================================

# Show detailed error messages
SHOW_ERRORS = True

# Print debug information
DEBUG_MODE = False
