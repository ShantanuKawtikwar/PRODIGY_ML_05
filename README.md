# FoodLens

FoodLens now deploys as a Vercel-friendly static app. The web interface performs lightweight food recognition directly in the browser with TensorFlow.js MobileNet and maps supported labels to calorie estimates. This removes the PyTorch serverless build that was failing on Vercel and keeps the deployed footprint well below the function size limit.

The original PyTorch training scripts are still included for local coursework use. They are no longer part of the deployed Vercel runtime.

## Why the old Vercel deployment failed

- `api/index.py` made Vercel build a Python Function.
- That function depended on pinned PyTorch CPU wheels from the PyTorch package index.
- Vercel could not install `torch==2.0.1+cpu` for its build platform.
- The repo also did not contain `best_food_model.pth`, so runtime inference would still fail even if the build passed.
- Shipping PyTorch plus weights is a poor fit for a small Vercel deployment target.

## Current deployment model

- Static frontend only
- No Python lambda
- No `uv sync` deployment step
- No bundled `.pth` model file
- In-browser inference through CDN-hosted TensorFlow.js assets

## Deploy to Vercel

1. Push this repository to GitHub.
2. Import the repository into Vercel.
3. Leave the project as a static deployment with no custom build command.
4. Deploy.

With this layout, Vercel should not try to install PyTorch or build a Python serverless function.

## Local ML workflow

Use the Python scripts only for local experimentation and training.

### Install local dependencies

```bash
pip install -r requirements-local.txt
```

### Configure Kaggle

1. Create a Kaggle API token from your Kaggle account settings.
2. Place `kaggle.json` in:
   - Windows: `C:\Users\<YourUsername>\.kaggle\kaggle.json`
   - Linux/macOS: `~/.kaggle/kaggle.json`

### Download Food-101

```bash
python download_data.py
```

### Train locally

```bash
python train.py
```

### Run local CLI prediction

```bash
python predict.py "path/to/your/image.jpg"
```

## Repo structure

- `index.html`, `styles.css`, `script.js`: static Vercel app
- `calorie_map.json`: calorie estimates used by the web UI
- `download_data.py`, `dataset.py`, `train.py`, `predict.py`: local PyTorch workflow
- `requirements-local.txt`: local-only Python dependencies
- `vercel.json`: minimal static Vercel config
