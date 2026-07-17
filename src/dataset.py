import kagglehub
import os
import shutil

def download_cotton_dataset():
    path = kagglehub.dataset_download("sabuktagin/dataset-for-cotton-leaf-disease-detection")
    print("Dataset downloaded to path:", path)
    return path
