import os
import requests

def download_file(url, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Downloading dataset from {url}...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Dataset downloaded to {output_path}")
    else:
        print(f"Failed to download dataset. Status code: {response.status_code}")

if __name__ == "__main__":
    # Download Zenodo dataset (record 5142664)
    dataset_url = "https://zenodo.org/record/5142664/files/maestro-v3.0.0-midi.zip?download=1"
    output_path = "data/maestro-v3.0.0-midi.zip"
    download_file(dataset_url, output_path)
