import pickle
import numpy as np
import requests
from tqdm import tqdm
import tarfile 
import deeplake



CIFAR_DATASET_URL = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
DOWNLAOD_FILENAME = "dataset.tar.gz"
PLACS_URL = "hub://activeloop/places205"


class DatasetDownloader:

    def download_item(url: str, path: str):
        try:
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            response.raise_for_status()

            # Open a file to write the content
            with open(path, "wb") as file:
                prog = tqdm(response.iter_content(chunk_size=8192),
                            desc="writing file",
                            total= total_size // 8192 + 1,
                            unit="KB")
                for chunk in prog:
                    if chunk:  
                        file.write(chunk)
                        prog.update()

            print(f"Downloaded successfully: {path}")
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
    
    def unzip(path: str):
        with tarfile.open(path) as file:
            file.extractall()
            print(f"Extracted {path}")

    def unpickle(path: str) -> np.ndarray:
        with open(path, "rb") as f:
            dict = pickle.load(f, encoding="bytes")
        return dict[b'data']
    
    def get_places():
        return deeplake.load("hub://activeloop/places205")
        
    


def main():
    # DatasetDownloader.download_item(CIFAR_DATASET_URL, DOWNLAOD_FILENAME)
    # DatasetDownloader.unzip(DOWNLAOD_FILENAME)
    # data: dict = DatasetDownloader.unpickle("cifar-100-python/train")
    # print(data.shape)
    DatasetDownloader.get_places()


if __name__ == "__main__":
    main()