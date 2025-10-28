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

# Hugging Face model ID or local path
# Popular options:
#   - "runwayml/stable-diffusion-v1-5" (default, balanced)
#   - "stabilityai/stable-diffusion-2-1" (improved quality)
#   - "prompthero/openjourney" (artistic style)
#   - "wavymulder/Analog-Diffusion" (analog photo style)
MODEL_ID = "runwayml/stable-diffusion-v1-5"

# Use safetensors format (faster and safer)
USE_SAFETENSORS = True

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

# Maximum values (safety limits)
MAX_STEPS = 150
MAX_GUIDANCE_SCALE = 30
MAX_DIMENSION = 1024
MAX_BATCH_SIZE = 10

# ============================================================================
# DEBUGGING
# ============================================================================

# Show detailed error messages
SHOW_ERRORS = True

# Print debug information
DEBUG_MODE = False
