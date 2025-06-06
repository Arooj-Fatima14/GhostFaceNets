import os
import shutil
from pathlib import Path
from tqdm import tqdm

# Define source and destination paths
source_dir = Path('/l/PathoGen/Face/New_folder/faces_webface_112x112_112x112_folders')
dest_dir = Path('/l/PathoGen/Face/New_folder/faces_umd_112x112_folders')

def move_folders():
    # Get list of folders in source directory
    folders = [f for f in source_dir.iterdir() if f.is_dir()]
    total_folders = len(folders)

    # Check if directories exist
    if not source_dir.exists():
        print(f"Error: Source directory {source_dir} does not exist.")
        return
    if not dest_dir.exists():
        print(f"Error: Destination directory {dest_dir} does not exist.")
        return

    print(f"Moving {total_folders} folders...")

    # Move folders with progress bar
    for folder in tqdm(folders, desc="Progress", unit="folder"):
        try:
            shutil.move(str(folder), str(dest_dir))
        except Exception as e:
            print(f"Failed to move {folder.name}: {str(e)}")

    print("Move operation completed.")

if __name__ == "__main__":
    move_folders()