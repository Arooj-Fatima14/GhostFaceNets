import sys
from pathlib import Path
import shutil
import time
from datetime import datetime

source_input = input("Enter the source directory path: ").strip()
dest_input = input("Enter the destination directory path: ").strip()
last_folder = int(input("Enter the last folder number: "))

source_path = Path(source_input)
dest_path = Path(dest_input)
next_folder_num = last_folder + 1

if not source_path.exists() or not source_path.is_dir():
    print(f"Invalid source path: {source_path}")
    sys.exit(1)

if not dest_path.exists() or not dest_path.is_dir():
    print(f"Invalid destination path: {dest_path}")
    sys.exit(1)

moved_count = 0
start_time = time.time()

while True:
    # Find the lowest numeric folder in source
    numeric_folders = [f for f in source_path.iterdir() if f.is_dir() and f.name.isdigit()]
    if not numeric_folders:
        break  # No folders left to move

    # Pick folder with smallest numeric name
    folder_to_move = min(numeric_folders, key=lambda f: int(f.name))
    new_name = str(next_folder_num)
    dest_folder = dest_path / new_name

    current_time = datetime.now().strftime("%H:%M:%S")
    elapsed = time.time() - start_time

    if dest_folder.exists():
        print(f"[{current_time} | +{elapsed:.2f}s] Skipping folder {new_name}: already exists")
    else:
        try:
            shutil.move(str(folder_to_move), str(dest_folder))
            moved_count += 1
            print(f"[{current_time} | +{elapsed:.2f}s] Moved folder {folder_to_move.name} to {new_name}")
            next_folder_num += 1
        except Exception as e:
            print(f"[{current_time} | +{elapsed:.2f}s] Error moving folder {folder_to_move.name}: {e}")

print(f"Completed: Moved {moved_count} folders from {last_folder + 1} to {next_folder_num - 1}")
