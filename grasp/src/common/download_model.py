import os
import gdown

def download_from_google_drive(file_id, destination):
    URL = "https://drive.google.com/uc?export=download"

    # Check if the file is already present
    if os.path.exists(destination):
        print(str(destination), " already exists. Skipping download.")
        return

    parent_dir = os.path.dirname(destination)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    gdown.download(f'https://drive.google.com/uc?id={file_id}', destination, quiet=False)
