# Quick Start Guide

Get your Stable Diffusion app running in 5 minutes!

## Prerequisites
- Windows 10/11
- Python 3.10 installed
- AMD GPU with updated drivers

## Installation (Choose One Method)

### Method 1: Using Setup Scripts (Easiest)

1. **Run Setup**:
   ```bash
   setup.bat
   ```
   This will create a virtual environment and install all dependencies.

2. **Start the App**:
   ```bash
   start.bat
   ```
   The web interface will open automatically!

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

3. **Test Installation** (Optional):
   ```bash
   python test_installation.py
   ```

4. **Run the App**:
   ```bash
   python app.py
   ```

## First Run

⏱️ **First launch will take 5-10 minutes** as it downloads the AI model (~4-5GB)

The model is cached, so subsequent launches are much faster!

## Using the App

1. **Enter a Prompt**: Describe what you want to see
   ```
   A beautiful sunset over mountains with a lake
   ```

2. **Adjust Settings** (optional):
   - Steps: 30 (good default)
   - Guidance Scale: 7.5 (good default)
   - Size: 512x512 (faster) or 768x768 (better quality)

3. **Click "Generate Images"**

4. **Wait**: Generation takes 30-60 seconds

5. **Save**: Click "Save Images" to save to `outputs/` folder

## Tips

- **Better Prompts**: Add details like "highly detailed", "4k", "photorealistic"
- **Negative Prompts**: Add "ugly, blurry, low quality" to avoid bad results
- **Batch Generation**: Set "Number of Images" to 4 to get multiple variations
- **Reproducibility**: Use a specific seed (not -1) to recreate images

## Customization

Edit `config.py` to change:
- Default GPU device
- Model to use
- Default settings
- Server port
- And more!

## Troubleshooting

### App won't start
```bash
# Test your installation
python test_installation.py

# Check GPU
python gpu_config.py
```

### Out of memory
- Use smaller dimensions (512x512 instead of 768x768)
- Generate fewer images at once
- Close other GPU apps

### Slow generation
- First generation is always slow (model loading)
- Reduce steps if needed (20-30 is usually enough)
- Check if GPU 1 is being used (Task Manager)

## What's Next?

- Read the full [README.md](README.md) for detailed documentation
- Try different models by editing `config.py`
- Experiment with different prompts and settings
- Check the example prompts in the UI

## Quick Reference

| File | Purpose |
|------|---------|
| `app.py` | Main application (run this!) |
| `config.py` | Settings (edit this to customize) |
| `setup.bat` | One-click setup |
| `start.bat` | One-click start |
| `test_installation.py` | Verify installation |
| `outputs/` | Your generated images |

## Getting Help

1. Check [README.md](README.md) Troubleshooting section
2. Run `python test_installation.py`
3. Check console for error messages

---

**Enjoy creating AI art!** 🎨
