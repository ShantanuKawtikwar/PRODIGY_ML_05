# Food Recognition & Calorie Estimation

This repository contains the necessary scripts to download the **Food-101** dataset from Kaggle, train a deep learning model (ResNet-50 via PyTorch) to recognize the 101 food categories, and run inference on new images to predict the food class and estimate its calorie content based on a static mapping.

## Setup Instructions

### 1. Install Dependencies

First, ensure you have Python 3.8+ installed. It is highly recommended to use a virtual environment.

```bash
pip install -r requirements.txt
```

### 2. Configure Kaggle API

To download the dataset directly via the Kaggle CLI, you must configure your API credentials:

1. Create a Kaggle account if you don't have one: https://www.kaggle.com/
2. Go to your Account Settings page (click your profile picture -> Settings).
3. Scroll down to the **API** section and click **"Create New Token"**. This will download a file named `kaggle.json`.
4. Place this file in your Kaggle configuration directory:
   - **Windows:** `C:\Users\<YourUsername>\.kaggle\kaggle.json`
   - **Linux/Mac:** `~/.kaggle/kaggle.json`

*(Important: Ensure the file has appropriate read permissions, e.g., `chmod 600 ~/.kaggle/kaggle.json` on Linux/Mac)*

### 3. Download the Dataset

Run the provided download script, which uses the Kaggle API to fetch and extract the `dansbecker/food-101` dataset into a `data/` folder in this directory.

```bash
python download_data.py
```
*Note: The dataset is approximately 5GB. This may take several minutes depending on your internet connection.*

### 4. Train the Model

To train the ResNet-50 transfer learning model, run:

```bash
python train.py
```
- By default, this trains for 5 epochs. You can adjust this directly in the script.
- The script automatically detects if a GPU (CUDA or MPS/Metal) is available and accelerates training.
- After training completes, the best model weights will be saved to `best_food_model.pth`.

### 5. Predict and Estimate Calories

To use the trained model to infer the food class and estimate calories from a new image, use:

```bash
python predict.py "path/to/your/food_image.jpg"
```

The script will scale/crop the image appropriately, run it through the network, look up the corresponding calorie estimation from `calorie_map.json`, and print out:
- Predicted Class
- Confidence Score (%)
- Estimated Calories (per 100g)

## Files Included

- `requirements.txt`: Python package dependencies.
- `download_data.py`: Script to fetch the Food-101 dataset via Kaggle.
- `calorie_map.json`: A dictionary mapping all 101 classes to their estimated caloric density (per 100g).
- `dataset.py`: PyTorch dataset class to load and augment images.
- `train.py`: Script to initialize ResNet-50 and train it. 
- `predict.py`: CLI inference tool.
