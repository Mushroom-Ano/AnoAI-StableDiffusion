# AnoAI - Stable Diffusion with DirectML

<div align="center">

**🎨 Professional AI Image Generation on AMD GPUs**

A powerful, user-friendly desktop application for generating stunning AI images using Stable Diffusion with DirectML support for AMD graphics cards on Windows.

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![DirectML](https://img.shields.io/badge/DirectML-AMD%20GPU-red.svg)](https://github.com/microsoft/DirectML)
[![Gradio](https://img.shields.io/badge/UI-Gradio-orange.svg)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Features](#-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Usage](#-usage) • [Documentation](#-documentation)

</div>

---

## 🌟 Highlights

- ⚡ **Optimized for AMD GPUs** - Leverages DirectML for hardware acceleration on AMD graphics cards
- 🎯 **Smart GPU Selection** - Automatically uses your discrete GPU (bypasses integrated graphics)
- 🖼️ **Beautiful Web UI** - Clean, intuitive interface built with Gradio
- 🚀 **Production Ready** - Robust error handling and user-friendly error messages
- 🔧 **Highly Configurable** - Easy-to-edit config file for all settings
- 📦 **Modular Architecture** - Clean code structure for easy customization

## ✨ Features

### Core Capabilities
- 🎨 **Text-to-Image Generation** - Create images from text descriptions
- 🖼️ **Image-to-Image Transformation** - Modify existing images with AI guidance
- 🔄 **Model Switching** - Load local models or choose from HuggingFace library
- 📊 **Batch Generation** - Generate multiple variations at once (1-10 images)
- 🎛️ **Full Control** - Adjust inference steps, guidance scale, dimensions, and seed
- 💾 **One-Click Save** - Download generated images with organized naming
- 🔁 **Reproducible Results** - Use seeds to recreate exact images

### Technical Features
- **DirectML Backend** - Native Windows GPU acceleration for AMD cards
- **GPU Device Forcing** - Ensures discrete GPU usage (Device 1: AMD RX 9070 XT)
- **Memory Optimized** - Attention slicing and automatic image resizing to prevent OOM errors
- **Smart Model Management** - Automatic discovery of local models and easy switching
- **Progress Tracking** - Real-time generation progress in the UI
- **Queue System** - Handles multiple requests gracefully

### User Experience
- **Preset Prompts** - Start generating immediately with example prompts
- **Negative Prompts** - Fine-tune results by specifying what to avoid
- **Error Recovery** - Graceful failure handling with helpful messages
- **Output Organization** - Auto-generated timestamped filenames

## 📋 System Requirements

| Component | Requirement |
|-----------|-------------|
| **OS** | Windows 10/11 (64-bit) |
| **Python** | 3.10 or higher |
| **GPU** | AMD GPU with DirectML support |
| **VRAM** | 6GB minimum for SD 1.5, 12GB+ recommended for SDXL |
| **RAM** | 16GB minimum, 32GB recommended for SDXL |
| **Storage** | ~5GB for model files + space for outputs |
| **Internet** | Required for initial model download |

**Tested Configuration:**
- AMD Radeon RX 9070 XT (16GB VRAM)
- Windows 11
- Python 3.10

**Important Note on DirectML:**
- DirectML is less memory-efficient than CUDA
- SDXL models may be limited to 512x512 or 768x768 even with 16GB VRAM
- SD 1.5 models work perfectly at all resolutions
- For better SDXL performance, consider WSL2 with ROCm

## 🚀 Quick Start

### One-Line Install (Windows)

```bash
# Clone the repository
git clone https://github.com/mushroom-ano/AnoAI-StableDiffusion.git
cd AnoAI-StableDiffusion

# Run the setup script
setup.bat

# Start the application
start.bat
```

The web interface will open automatically at `http://127.0.0.1:7860`

### First Generation

1. Enter a prompt: `"A majestic mountain landscape at sunset, 4k, detailed"`
2. Click **"🎨 Generate Images"**
3. Wait 30-60 seconds for your first image
4. Click **"💾 Save Images"** to download

## 📦 Installation

### Method 1: Automated Setup (Recommended)

1. **Download or Clone** this repository:
   ```bash
   git clone https://github.com/mushroom-ano/AnoAI-StableDiffusion.git
   cd AnoAI-StableDiffusion
   ```

2. **Run Setup**:
   ```bash
   setup.bat
   ```
   This creates a virtual environment and installs all dependencies.

3. **Start the App**:
   ```bash
   start.bat
   ```

### Method 2: Manual Setup

1. **Create Virtual Environment**:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Test Installation** (optional):
   ```bash
   python test_installation.py
   ```

4. **Run Application**:
   ```bash
   python app.py
   ```

### First Run Notes

⏱️ **First launch takes 5-10 minutes** as it downloads the Stable Diffusion model (~4-5GB)

The model is cached locally, so subsequent launches are much faster (5-10 seconds).

## 🎯 Usage

### Text-to-Image Generation

1. **Enter Your Prompt**
   ```
   A serene Japanese garden with cherry blossoms, koi pond, soft lighting, highly detailed
   ```

2. **Set Negative Prompt** (optional)
   ```
   blurry, low quality, distorted, ugly, bad anatomy
   ```

3. **Adjust Settings** (optional)
   - **Steps**: 20-30 for quick iterations, 40-50 for best quality
   - **Guidance Scale**: 7-12 for most prompts
   - **Size**: 512x512 (fast) or 768x768 (better quality)
   - **Seed**: -1 for random, or specific number to reproduce

4. **Generate & Save**
   - Click **"🎨 Generate Images"**
   - Wait for generation (30-60 seconds)
   - Click **"💾 Save Images"** to download

### Image-to-Image Transformation

Transform existing images with AI guidance:

1. **Upload an Image**
   - Click the upload area or drag & drop
   - Supported formats: PNG, JPG, JPEG

2. **Enter Your Transformation Prompt**
   ```
   Transform into an oil painting style, impressionist, vibrant colors
   ```

3. **Adjust Strength** (0.0 - 1.0)
   - **0.3-0.5**: Subtle changes, keeps original composition
   - **0.6-0.8**: Major transformations
   - **0.9-1.0**: Almost complete recreation

4. **Set Max Image Size** (optional)
   - Default: 768px (prevents OOM errors)
   - Larger images are automatically resized while maintaining aspect ratio

5. **Generate & Save**
   - Images are automatically adjusted to be divisible by 8
   - Generation time: 30-90 seconds depending on size

### Switching Models

Load different models without restarting:

1. **Open Model Settings** accordion (above the tabs)
2. **Select a Model**:
   - 📁 Local models from `models/` directory
   - ☁️ HuggingFace models (downloaded on first use)
3. **Click "📥 Load Model"**
4. Wait 30-60 seconds for model to load
5. Start generating with the new model

**Adding Local Models**:
- Place models in the `models/` directory
- Supported formats: Diffusers folders, `.safetensors`, `.ckpt` files
- See `models/README.md` for detailed instructions

### Pro Tips

💡 **Better Prompts**:
- Be specific and descriptive
- Include style keywords: "photorealistic", "oil painting", "digital art"
- Add quality tags: "highly detailed", "4k", "masterpiece"
- Specify lighting: "golden hour", "studio lighting", "dramatic lighting"

💡 **Image-to-Image**:
- Start with strength 0.5-0.7 for best results
- Lower strength preserves more of the original image
- Use descriptive prompts about the desired style/changes
- Large images are automatically resized to prevent memory issues

💡 **Performance**:
- First generation is always slower (model loading)
- 512x512 @ 30 steps: ~30-45 seconds
- 768x768 @ 50 steps: ~90-120 seconds
- Batch generation is more efficient than one-by-one
- Model switching takes 30-60 seconds

💡 **Reproducibility**:
- Copy the seed from successful generations
- Use same seed + settings to recreate similar images
- Small prompt changes with same seed = controlled variations

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# GPU Selection
GPU_DEVICE_ID = 1  # 0 = integrated GPU, 1 = discrete GPU

# Model Configuration
MODELS_DIR = "models"  # Directory for local models
MODEL_ID = "runwayml/stable-diffusion-v1-5"  # Default model on startup
ENABLE_MODEL_SWITCHING = True  # Allow model switching in UI

# Default Settings
DEFAULT_STEPS = 30
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 512

# Image-to-Image Settings
IMG2IMG_MAX_DIMENSION = 768  # Maximum size before auto-resize
IMG2IMG_DEFAULT_STRENGTH = 0.75  # Default transformation strength

# Server Settings
SERVER_PORT = 7860
SHARE_PUBLICLY = False  # Set True to create public URL
```

### Using Different Models

**Option 1: Via UI** (Recommended)
- Use the Model Settings accordion to switch models without restarting

**Option 2: Via Config**
```python
MODEL_ID = "stabilityai/stable-diffusion-2-1"  # SD 2.1, improved quality
MODEL_ID = "prompthero/openjourney"  # Artistic style
MODEL_ID = "wavymulder/Analog-Diffusion"  # Analog photo style
MODEL_ID = "models/my-custom-model"  # Local model
```

## 📁 Project Structure

```
AnoAI-StableDiffusion/
├── app.py                 # Main application entry point
├── gpu_config.py          # DirectML GPU configuration & device selection
├── sd_pipeline.py         # Stable Diffusion pipeline wrapper (txt2img & img2img)
├── model_manager.py       # Model discovery and management
├── config.py              # User-configurable settings
├── requirements.txt       # Python dependencies
├── setup.bat              # Windows automated setup script
├── start.bat              # Windows app launcher
├── test_installation.py   # Installation verification script
├── .gitignore             # Git ignore rules
├── README.md              # This file
├── QUICKSTART.md          # Quick reference guide
├── models/                # Local models directory
│   └── README.md          # Instructions for adding models
└── outputs/               # Generated images (auto-created)
```

## 🔧 Troubleshooting

### App Won't Start

```bash
# Verify installation
python test_installation.py

# Check GPU detection
python gpu_config.py
```

### Out of Memory Errors

**For SD 1.5 Models:**
- Reduce image dimensions (512x512 instead of 768x768)
- Generate fewer images per batch
- Close other GPU-intensive applications

**For SDXL/Pony Models:**
- DirectML uses more VRAM than CUDA for SDXL
- Even with 16GB VRAM, you may be limited to 768x768 or lower
- Sequential CPU offload is automatically enabled (trades speed for memory)
- Expect 2-3x slower generation with SDXL models
- **Recommended**: Use 512x512 or 768x768 for SDXL on DirectML
- **Alternative**: Use SD 1.5 models which work perfectly at higher resolutions

### Slow Generation

- First generation is always slow (model loading)
- Reduce inference steps (20-30 is usually sufficient)
- Check Task Manager to verify GPU 1 is being used
- Ensure GPU drivers are up to date

### Generation Fails

- Check console for detailed error messages
- Verify prompt doesn't have invalid characters
- Ensure dimensions are divisible by 8
- Try reducing batch size to 1

## 🛠️ Advanced Usage

### Command Line Arguments

```bash
# Use different device
set TORCH_DIRECTML_DEVICE_INDEX=0
python app.py

# Custom port
# Edit config.py: SERVER_PORT = 7861
python app.py
```

### Adding Custom Features

The modular architecture makes it easy to extend:

- **New schedulers**: Edit `sd_pipeline.py`
- **Custom models**: Add to `models/` directory or `model_manager.py`
- **Custom UI elements**: Modify `create_ui()` in `app.py`
- **Pipeline modifications**: Extend `StableDiffusionGenerator` class

## 📚 Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - Quick reference guide
- **[models/README.md](models/README.md)** - How to add and use local models
- **[GPU Configuration](gpu_config.py)** - How GPU selection works
- **[Pipeline Details](sd_pipeline.py)** - Stable Diffusion pipeline implementation
- **[Model Manager](model_manager.py)** - Model discovery and switching

### Example Prompts

**Photorealistic**:
```
A professional photograph of a steaming cup of coffee on a wooden table,
morning sunlight streaming through window, shallow depth of field, 4k, highly detailed
```

**Artistic**:
```
A mystical forest with glowing mushrooms, ethereal lighting, fantasy art style,
vibrant colors, digital painting, trending on artstation
```

**Portrait**:
```
Portrait of a wise old wizard with long white beard, magical robes,
staff with glowing crystal, fantasy art, highly detailed face, dramatic lighting
```

**Architecture**:
```
Modern minimalist house with large glass windows, surrounded by nature,
architectural photography, golden hour lighting, 8k, ultra realistic
```

**Image-to-Image**:
```
Upload: A photo of a cat
Prompt: Transform into a watercolor painting, soft colors, artistic
Strength: 0.7
```

## ⚠️ Known Limitations

### DirectML Memory Efficiency
- **SDXL Models**: DirectML's memory management is less efficient than CUDA
  - Limited to 512x512 or 768x768 on most systems, even with 16GB VRAM
  - Generation is 2-3x slower due to sequential CPU offload
  - Consider using SD 1.5 models for better performance
- **SD 1.5 Models**: Work perfectly at all resolutions (512x512, 768x768, 1024x1024)

### NSFW Safety Checker
- Disabled by default to prevent false positives
- Can be re-enabled in `config.py` by setting `DISABLE_SAFETY_CHECKER = False`

### Seed Reproducibility
- Seeds work but may not be fully deterministic on DirectML
- Results may vary slightly between generations with the same seed

### Model Format Support
- **Diffusers format**: Full support (recommended)
- **Checkpoint files** (.safetensors/.ckpt): Supported via `from_single_file()`
- Requires recent version of `diffusers` library

### Alternative: ROCm on WSL2
For better SDXL performance on AMD GPUs:
- Install WSL2 (Windows Subsystem for Linux)
- Use ROCm instead of DirectML
- Significantly better memory efficiency and speed
- Requires Linux environment setup

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- [x] Image-to-image generation
- [x] Multiple model switching in UI
- [x] SDXL/Pony model support
- [ ] Inpainting support
- [ ] LoRA support
- [ ] ControlNet support
- [ ] Prompt history/favorites
- [ ] Upscaling integration
- [ ] Advanced scheduler options
- [ ] WSL2/ROCm setup guide

## 📄 License

This project is licensed under the MIT License - see below for details.

### Third-Party Licenses

- **Stable Diffusion**: [CreativeML Open RAIL-M License](https://huggingface.co/spaces/CompVis/stable-diffusion-license)
- **Diffusers**: Apache 2.0 License
- **Gradio**: Apache 2.0 License
- **DirectML**: MIT License

## 🙏 Acknowledgments

- **Stability AI** - For Stable Diffusion
- **Hugging Face** - For the Diffusers library
- **Microsoft** - For DirectML
- **Gradio Team** - For the excellent UI framework

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/mushroom-ano/AnoAI-StableDiffusion/issues)
- **Discussions**: [GitHub Discussions](https://github.com/mushroom-ano/AnoAI-StableDiffusion/discussions)

## 🔖 Version

**Version**: 2.0.0
**Last Updated**: January 2025
**Tested On**: Windows 11, AMD RX 9070 XT (16GB VRAM), Python 3.10

### Changelog

**v2.0.0** (Current)
- Added image-to-image transformation
- Added model switching UI (local models + HuggingFace)
- Added SDXL/Pony Diffusion model support
- Added automatic model architecture detection
- Added aggressive memory optimizations for SDXL
- Added memory usage monitoring
- Disabled NSFW safety checker by default
- Added support for .safetensors and .ckpt checkpoint files
- Documented DirectML limitations with SDXL models

**v1.0.0** (Initial Release)
- Text-to-image generation with SD 1.5
- DirectML GPU support for AMD
- Gradio web UI
- Batch generation
- Basic error handling

---

<div align="center">

**⭐ If you find this project useful, please consider giving it a star!**

Made with ❤️ by [mushroom-ano](https://github.com/mushroom-ano)

</div>
