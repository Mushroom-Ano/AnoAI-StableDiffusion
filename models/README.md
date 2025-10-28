# Models Directory

Place your Stable Diffusion models here to use them locally without re-downloading.

## Supported Formats

- **Diffusers Format** (folder with model files) - Recommended, fastest loading
- **SafeTensors** (.safetensors files) - Single-file checkpoints
- **Checkpoint** (.ckpt files) - Legacy checkpoint format

## Supported Model Architectures

- **Stable Diffusion 1.5** - Standard SD models (~4 GB file size)
  - VRAM requirement: 4-6 GB
  - Recommended resolution: 512x512 or 768x768
  - Generation time: ~30-60 seconds

- **SDXL/Pony Diffusion** - Higher quality SDXL-based models (~6-7 GB file size)
  - VRAM requirement: 8-12 GB (with optimizations enabled)
  - Recommended resolution: 768x768 or 1024x1024
  - Generation time: ~60-120 seconds

**Auto-Detection**: The application automatically detects whether a model is SD 1.5 or SDXL-based by:
- File size (SDXL models are typically >5.5 GB)
- Model name (contains "xl", "sdxl", or "pony")

### SDXL Memory Optimizations

**DirectML VRAM Issue**: DirectML is less memory-efficient than CUDA. Even with 16GB VRAM, SDXL models may struggle at 1024x1024 resolution.

The application automatically enables aggressive optimizations for SDXL:
- **Max Attention Slicing**: Maximum memory savings for attention operations
- **VAE Slicing**: Reduces VRAM for image encoding/decoding
- **VAE Tiling**: Processes large images in tiles
- **Sequential CPU Offload**: Most aggressive memory saving (trades speed for memory)
  - Only loads model components when needed
  - Significantly slower (2-3x) but allows higher resolutions
  - Components shuttle between RAM and VRAM as needed

**Performance Impact**:
- With optimizations: Slower generation but supports 768x768 or 1024x1024
- Without optimizations: Faster but limited to 512x512 on DirectML

If you still get out-of-memory errors with SDXL:
1. Start with 512x512 and gradually increase resolution
2. Generate 1 image at a time (not batches)
3. Close other GPU-intensive applications
4. Consider using SD 1.5 models (no memory issues, still great quality)

**SD 1.5 vs SDXL on DirectML**:
- SD 1.5: Works perfectly at 512x512 and 768x768, fast
- SDXL: Better quality but memory-constrained, slower with DirectML

**Note**: Checkpoint files (.safetensors/.ckpt) are automatically loaded using `from_single_file()`. This requires a recent version of the `diffusers` library.

## Directory Structure

```
models/
├── stable-diffusion-v1-5/          # Diffusers format (folder)
│   ├── model_index.json
│   ├── vae/
│   ├── text_encoder/
│   ├── unet/
│   └── ...
├── openjourney-v4/                 # Another model
├── my-custom-model.safetensors     # SafeTensors format
└── another-model.ckpt              # Checkpoint format
```

## How to Add Models

### Option 1: Download from Hugging Face

```bash
# Install git-lfs if you haven't
git lfs install

# Clone a model repository
cd models
git clone https://huggingface.co/runwayml/stable-diffusion-v1-5
```

### Option 2: Copy Existing Models

If you already have models downloaded:

```bash
# Copy from HuggingFace cache
cp -r ~/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/[hash]/* models/stable-diffusion-v1-5/

# Or copy from another location
cp -r /path/to/your/model models/my-model-name/
```

### Option 3: Place Model Files

Just copy .safetensors or .ckpt files directly:

```bash
cp /path/to/model.safetensors models/
```

## Popular Models to Try

- **stable-diffusion-v1-5** - `runwayml/stable-diffusion-v1-5`
- **stable-diffusion-2-1** - `stabilityai/stable-diffusion-2-1`
- **OpenJourney** - `prompthero/openjourney`
- **Analog Diffusion** - `wavymulder/Analog-Diffusion`
- **Dreamlike Photoreal** - `dreamlike-art/dreamlike-photoreal-2.0`

## Model Selection

1. Place models in this directory
2. Restart the application
3. Select your model from the dropdown in the UI
4. Click "Load Model" to switch

## Notes

- Models are loaded into memory, so only one can be active at a time
- Switching models takes 30-60 seconds
- Default model from config.py will load on startup
