import os
import requests
import zipfile
import io

def download_and_extract_zip(url, extract_to='data/bread-midi-dataset'):
    os.makedirs(extract_to, exist_ok=True)
    print(f"Downloading dataset from {url}...")
    response = requests.get(url)
    if response.status_code == 200:
        print("Download complete. Extracting files...")
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Dataset extracted to {extract_to}")
    else:
        print(f"Failed to download dataset. Status code: {response.status_code}")

if __name__ == "__main__":
    dataset_url = "https://huggingface.co/datasets/breadlicker45/bread-midi-dataset/resolve/main/bread-midi-dataset.zip"
    download_and_extract_zip(dataset_url)
