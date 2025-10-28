# Stable Diffusion with DirectML on Windows

A clean, user-friendly web application for generating AI images using Stable Diffusion on AMD GPUs with DirectML support on Windows.

## Features

- **DirectML GPU Support**: Optimized for AMD GPUs (specifically tested on AMD RX 9070 XT)
- **Force GPU Selection**: Automatically uses GPU device 1 (discrete GPU) instead of integrated GPU
- **Clean Web UI**: Built with Gradio for an intuitive user experience
- **Batch Generation**: Generate multiple images from a single prompt
- **Full Control**: Adjust steps, guidance scale, image dimensions, and seed
- **Progress Tracking**: Real-time progress updates during generation
- **Image Management**: Save and download generated images
- **Error Handling**: User-friendly error messages and graceful failure handling
- **Modular Design**: Easy to extend and customize

## System Requirements

- **OS**: Windows 10/11
- **Python**: 3.10 (recommended)
- **GPU**: AMD GPU with DirectML support (tested on AMD RX 9070 XT)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: ~5GB for model files

## Project Structure

```
AnoAI/
├── app.py              # Main application with Gradio UI
├── gpu_config.py       # GPU configuration and DirectML device selection
├── sd_pipeline.py      # Stable Diffusion pipeline wrapper
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── outputs/           # Generated images (created automatically)
```

## Installation

### Step 1: Prerequisites

1. **Update GPU Drivers**: Ensure you have the latest AMD GPU drivers installed
   - Download from: https://www.amd.com/en/support

2. **Install Python 3.10**: Download from https://www.python.org/downloads/
   - Make sure to check "Add Python to PATH" during installation

3. **Verify Python Installation**:
   ```bash
   python --version
   # Should show Python 3.10.x
   ```

### Step 2: Create Virtual Environment (Recommended)

Open Command Prompt or PowerShell in the project directory:

```bash
# Create virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate

# You should see (venv) in your prompt
```

### Step 3: Install Dependencies

```bash
# Make sure you're in the project directory with requirements.txt
pip install -r requirements.txt
```

This will install:
- `torch-directml`: DirectML backend for PyTorch
- `diffusers`: Hugging Face diffusers library for Stable Diffusion
- `transformers`: Required by diffusers
- `gradio`: Web UI framework
- `pillow`: Image processing
- Other utilities

**Note**: The first installation may take several minutes as it downloads all dependencies.

### Step 4: Test GPU Configuration

Before running the full application, test that your GPU is properly detected:

```bash
python gpu_config.py
```

You should see output like:
```
============================================================
GPU CONFIGURATION
============================================================
Device Id           : 1
Device Name         : DirectML Device 1
Backend             : DirectML (Windows)
Device Object       : privateuseone:0
============================================================
Using GPU Device 1 (AMD RX 9070 XT)
Device 0 (integrated GPU) will be ignored
============================================================
```

## Usage

### Starting the Application

1. **Activate Virtual Environment** (if not already activated):
   ```bash
   venv\Scripts\activate
   ```

2. **Run the Application**:
   ```bash
   python app.py
   ```

3. **First Run**: The first time you run the app, it will:
   - Download the Stable Diffusion model from Hugging Face (~4-5GB)
   - Load the model into GPU memory
   - This may take 5-10 minutes depending on your internet speed

4. **Access the UI**: The app will automatically open in your browser at:
   - http://127.0.0.1:7860

### Using the Web Interface

#### Prompt Section
- **Prompt**: Describe what you want to see in detail
  - Example: "A serene mountain landscape with a crystal clear lake reflecting the golden sunset, detailed, 4k, photorealistic"
- **Negative Prompt**: Describe what you want to avoid
  - Example: "ugly, blurry, low quality, distorted, deformed"

#### Settings
- **Steps** (20-50 recommended):
  - More steps = better quality but slower generation
  - 20-30 steps: Good for quick iterations
  - 40-50 steps: High quality results

- **Guidance Scale** (7-12 recommended):
  - How closely the AI follows your prompt
  - Lower (5-7): More creative/varied results
  - Higher (10-15): More literal interpretation

- **Width/Height** (512x512 default):
  - Must be divisible by 8
  - 512x512: Fast, good quality
  - 768x768: Higher quality, slower
  - 1024x1024: Highest quality, slowest

- **Seed**:
  - Use `-1` for random seed (different result each time)
  - Use a specific number (e.g., `42`) to reproduce the same image

- **Number of Images** (1-10):
  - Generate multiple variations at once
  - Useful for exploring different interpretations

#### Buttons
- **Generate Images**: Start the generation process
- **Save Images**: Save the currently displayed images to the `outputs/` folder
- **Clear**: Clear the gallery and status message

### Example Prompts

**Landscape**:
```
Prompt: A majestic mountain range at sunrise, golden hour lighting, mist in valleys, photorealistic, 8k, highly detailed
Negative: blurry, low quality, oversaturated
```

**Portrait**:
```
Prompt: Portrait of a wise old wizard with long white beard, magical lighting, fantasy art, detailed, digital painting
Negative: ugly, deformed, cartoon, anime
```

**Architecture**:
```
Prompt: Modern minimalist house with large windows, surrounded by forest, architectural photography, natural lighting
Negative: cluttered, busy, dark, gloomy
```

**Sci-Fi**:
```
Prompt: Futuristic cyberpunk city at night, neon lights, flying cars, rain, cinematic, blade runner style
Negative: daylight, bright, low detail
```

## GPU Configuration

### Understanding Device Selection

- **Device 0**: Typically the integrated GPU (CPU graphics)
- **Device 1**: Typically the discrete GPU (your AMD RX 9070 XT)

This application is configured to use **Device 1** by default.

### How GPU Selection Works

The GPU selection happens in `gpu_config.py`:

```python
# Force DirectML to use device 1 (discrete GPU)
os.environ['TORCH_DIRECTML_DEVICE_INDEX'] = str(device_id)
device = torch_directml.device(device_id)
```

### Changing GPU Device

If you need to use a different device:

1. **Edit `app.py`**, line ~305:
   ```python
   app = StableDiffusionApp(
       device_id=1,  # Change this number
       model_id="runwayml/stable-diffusion-v1-5"
   )
   ```

2. Or set environment variable before running:
   ```bash
   set TORCH_DIRECTML_DEVICE_INDEX=0
   python app.py
   ```

## Customization

### Using Different Models

You can use any Stable Diffusion model from Hugging Face:

**Edit `app.py`**, line ~306:
```python
app = StableDiffusionApp(
    device_id=1,
    model_id="stabilityai/stable-diffusion-2-1"  # Change this
)
```

Popular models:
- `runwayml/stable-diffusion-v1-5` (default, balanced)
- `stabilityai/stable-diffusion-2-1` (improved quality)
- `prompthero/openjourney` (artistic style)
- `wavymulder/Analog-Diffusion` (analog photo style)

### Adding Features

The modular structure makes it easy to add features:

- **`gpu_config.py`**: Modify GPU configuration and device management
- **`sd_pipeline.py`**: Add new generation methods or post-processing
- **`app.py`**: Extend the UI with new controls or features

Example: Add image-to-image generation in `sd_pipeline.py`
Example: Add style presets in `app.py`

## Troubleshooting

### Issue: "torch-directml is not installed"
```bash
pip install torch-directml
```

### Issue: "Failed to initialize DirectML device"
1. Update your AMD GPU drivers
2. Check if your GPU supports DirectML:
   ```bash
   python gpu_config.py
   ```
3. Try device 0 instead:
   ```python
   device_id=0
   ```

### Issue: Model download fails
1. Check your internet connection
2. Hugging Face may be slow - be patient
3. Try downloading manually:
   ```bash
   from diffusers import StableDiffusionPipeline
   StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
   ```

### Issue: Out of memory error
1. Reduce image dimensions (e.g., 512x512 instead of 768x768)
2. Generate fewer images at once
3. Close other GPU-intensive applications

### Issue: Generation is very slow
1. This is normal for the first generation (model loading)
2. Subsequent generations should be faster
3. Reduce steps if needed (20-30 is usually enough)
4. Check Task Manager to ensure GPU 1 is being used

### Issue: Images look low quality
1. Increase steps (try 40-50)
2. Adjust guidance scale (try 7-12)
3. Improve your prompt with more details
4. Add quality keywords: "highly detailed", "4k", "photorealistic"

### Issue: Web UI doesn't open
1. Check the console for the actual URL
2. Manually open: http://127.0.0.1:7860
3. Check if port 7860 is already in use
4. Change port in `app.py`:
   ```python
   demo.launch(server_port=7861)
   ```

## Performance Tips

1. **First Generation**: Always slower due to model loading (~1-2 minutes)
2. **Subsequent Generations**: Much faster (~30-60 seconds for 512x512 @ 30 steps)
3. **Batch Generation**: More efficient than generating one at a time
4. **Memory Management**: Close the app when done to free GPU memory

## Output Files

Generated images are saved to:
```
AnoAI/outputs/
├── prompt_name_20250128_143022_1.png
├── prompt_name_20250128_143022_2.png
└── ...
```

File naming format: `{prompt}_{timestamp}_{number}.png`

## Advanced Configuration

### Enable Model Caching
Models are cached by Hugging Face in:
```
C:\Users\{username}\.cache\huggingface\hub
```

To use a different cache location:
```bash
set HF_HOME=D:\Models\HuggingFace
python app.py
```

### Memory Optimization
In `sd_pipeline.py`, you can enable additional optimizations:
```python
# Already enabled in the code:
self.pipeline.enable_attention_slicing()

# You can also try:
self.pipeline.enable_vae_slicing()  # Reduces VRAM usage
```

### Share Your UI Publicly
In `app.py`, change:
```python
demo.launch(
    share=True  # Creates a public URL for 72 hours
)
```

## License

This project uses:
- Stable Diffusion: CreativeML Open RAIL-M License
- Diffusers library: Apache 2.0 License
- Gradio: Apache 2.0 License

## Credits

- **Stable Diffusion**: CompVis, Stability AI, and RunwayML
- **Diffusers**: Hugging Face
- **DirectML**: Microsoft
- **Gradio**: Gradio Team

## Support

For issues and questions:
1. Check the Troubleshooting section above
2. Verify GPU configuration with `python gpu_config.py`
3. Check console output for detailed error messages
4. Ensure all dependencies are installed correctly

## Future Enhancements

Possible additions:
- [ ] Image-to-image generation
- [ ] Inpainting support
- [ ] Style presets
- [ ] Prompt templates
- [ ] History/Gallery view
- [ ] Model switcher in UI
- [ ] Advanced scheduler options
- [ ] LoRA support
- [ ] Upscaling integration

## Version

**Version**: 1.0.0
**Last Updated**: January 2025
**Tested On**: Windows 11, AMD RX 9070 XT, Python 3.10

---

Enjoy creating AI art with Stable Diffusion! 🎨
