import os
import subprocess
import zipfile
import shutil

def download_food101():
    print("Initializing Food-101 Dataset Download via Kaggle API...")
    
    # Check if dataset already exists
    if os.path.exists('data/food-101'):
        print("Dataset 'data/food-101' already exists. Skipping download.")
        return

    os.makedirs('data', exist_ok=True)
    
    # Run the kaggle download command
    try:
        print("Downloading dansbecker/food-101 dataset (~5GB). This may take a while...")
        subprocess.run(
            ['kaggle', 'datasets', 'download', '-d', 'dansbecker/food-101', '-p', 'data'],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error downloading dataset. Ensure Kaggle API is configured correctly. Details: {e}")
        print("Required: Place kaggle.json in ~/.kaggle/ (Linux/Mac) or C:\\Users\\<Username>\\.kaggle\\ (Windows)")
        return
    except FileNotFoundError:
        print("Kaggle CLI not found. Please ensure 'kaggle' is installed and in your PATH.")
        return

    # Extract the zip file
    zip_path = os.path.join('data', 'food-101.zip')
    if os.path.exists(zip_path):
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall('data')
        
        print("Cleaning up zip file...")
        os.remove(zip_path)
        
        print("Download and extraction complete! Dataset is located at 'data/food-101'.")
    else:
        print("Download failed: Zip file not found.")

if __name__ == '__main__':
    download_food101()
