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
- 📊 **Batch Generation** - Generate multiple variations at once (1-10 images)
- 🎛️ **Full Control** - Adjust inference steps, guidance scale, dimensions, and seed
- 💾 **One-Click Save** - Download generated images with organized naming
- 🔄 **Reproducible Results** - Use seeds to recreate exact images

### Technical Features
- **DirectML Backend** - Native Windows GPU acceleration for AMD cards
- **GPU Device Forcing** - Ensures discrete GPU usage (Device 1: AMD RX 9070 XT)
- **Memory Optimized** - Attention slicing for efficient VRAM usage
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
| **RAM** | 8GB minimum, 16GB recommended |
| **Storage** | ~5GB for model files + space for outputs |
| **Internet** | Required for initial model download |

**Tested Configuration:**
- AMD Radeon RX 9070 XT (Device 1)
- Windows 11
- Python 3.10

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

### Basic Workflow

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

### Pro Tips

💡 **Better Prompts**:
- Be specific and descriptive
- Include style keywords: "photorealistic", "oil painting", "digital art"
- Add quality tags: "highly detailed", "4k", "masterpiece"
- Specify lighting: "golden hour", "studio lighting", "dramatic lighting"

💡 **Performance**:
- First generation is always slower (model loading)
- 512x512 @ 30 steps: ~30-45 seconds
- 768x768 @ 50 steps: ~90-120 seconds
- Batch generation is more efficient than one-by-one

💡 **Reproducibility**:
- Copy the seed from successful generations
- Use same seed + settings to recreate similar images
- Small prompt changes with same seed = controlled variations

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# GPU Selection
GPU_DEVICE_ID = 1  # 0 = integrated GPU, 1 = discrete GPU

# Model Selection
MODEL_ID = "runwayml/stable-diffusion-v1-5"  # Change to other models

# Default Settings
DEFAULT_STEPS = 30
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 512

# Server Settings
SERVER_PORT = 7860
SHARE_PUBLICLY = False  # Set True to create public URL
```

### Using Different Models

Popular alternatives:
```python
MODEL_ID = "stabilityai/stable-diffusion-2-1"  # SD 2.1, improved quality
MODEL_ID = "prompthero/openjourney"  # Artistic style
MODEL_ID = "wavymulder/Analog-Diffusion"  # Analog photo style
```

## 📁 Project Structure

```
AnoAI-StableDiffusion/
├── app.py                 # Main application entry point
├── gpu_config.py          # DirectML GPU configuration & device selection
├── sd_pipeline.py         # Stable Diffusion pipeline wrapper
├── config.py              # User-configurable settings
├── requirements.txt       # Python dependencies
├── setup.bat              # Windows automated setup script
├── start.bat              # Windows app launcher
├── test_installation.py   # Installation verification script
├── .gitignore             # Git ignore rules
├── README.md              # This file
├── QUICKSTART.md          # Quick reference guide
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

- Reduce image dimensions (512x512 instead of 768x768)
- Generate fewer images per batch
- Close other GPU-intensive applications
- Enable VAE slicing in `sd_pipeline.py`

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
- **Image-to-image**: Extend `StableDiffusionGenerator` class
- **Custom UI elements**: Modify `create_ui()` in `app.py`
- **Model presets**: Add to `config.py`

## 📚 Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - Quick reference guide
- **[GPU Configuration](gpu_config.py)** - How GPU selection works
- **[Pipeline Details](sd_pipeline.py)** - Stable Diffusion pipeline implementation

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

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- [ ] Image-to-image generation
- [ ] Inpainting support
- [ ] LoRA support
- [ ] Multiple model switching in UI
- [ ] Prompt history/favorites
- [ ] Upscaling integration
- [ ] Advanced scheduler options

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

**Version**: 1.0.0
**Last Updated**: January 2025
**Tested On**: Windows 11, AMD RX 9070 XT, Python 3.10

---

<div align="center">

**⭐ If you find this project useful, please consider giving it a star!**

Made with ❤️ by [mushroom-ano](https://github.com/mushroom-ano)

</div>
