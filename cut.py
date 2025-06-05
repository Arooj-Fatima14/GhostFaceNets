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
    try:
        # Use iterator to avoid scanning the full directory
        folder_iter = source_path.iterdir()
        for folder in folder_iter:
            if folder.is_dir() and folder.name.isdigit():
                new_name = str(next_folder_num)
                dest_folder = dest_path / new_name

                current_time = datetime.now().strftime("%H:%M:%S")
                elapsed = time.time() - start_time

                try:
                    shutil.move(str(folder), str(dest_folder))
                    print(f"[{current_time} | +{elapsed:.2f}s] Moved folder {folder.name} to {new_name}")
                    moved_count += 1
                    next_folder_num += 1
                except Exception as e:
                    print(f"[{current_time} | +{elapsed:.2f}s] Error moving {folder.name}: {e}")
                break  # Move one folder only, then restart fresh
        else:
            # No valid folders found, exit
            break
    except StopIteration:
        break

print(f"✅ Completed: Moved {moved_count} folders from {last_folder + 1} to {next_folder_num - 1}")
