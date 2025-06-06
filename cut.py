# import sys
# from pathlib import Path
# import shutil
# import time
# from datetime import datetime

# source_input = input("Enter the source directory path: ").strip()
# dest_input = input("Enter the destination directory path: ").strip()
# last_folder = int(input("Enter the last folder number: "))

# source_path = Path(source_input)
# dest_path = Path(dest_input)
# next_folder_num = last_folder + 1

# if not source_path.exists() or not source_path.is_dir():
#     print(f"Invalid source path: {source_path}")
#     sys.exit(1)

# if not dest_path.exists() or not dest_path.is_dir():
#     print(f"Invalid destination path: {dest_path}")
#     sys.exit(1)

# moved_count = 0
# start_time = time.time()

# while True:
#     try:
#         # Use iterator to avoid scanning the full directory
#         folder_iter = source_path.iterdir()
#         for folder in folder_iter:
#             if folder.is_dir() and folder.name.isdigit():
#                 new_name = str(next_folder_num)
#                 dest_folder = dest_path / new_name

#                 current_time = datetime.now().strftime("%H:%M:%S")
#                 elapsed = time.time() - start_time

#                 try:
#                     shutil.move(str(folder), str(dest_folder))
#                     print(f"[{current_time} | +{elapsed:.2f}s] Moved folder {folder.name} to {new_name}")
#                     moved_count += 1
#                     next_folder_num += 1
#                 except Exception as e:
#                     print(f"[{current_time} | +{elapsed:.2f}s] Error moving {folder.name}: {e}")
#                 break  # Move one folder only, then restart fresh
#         else:
#             # No valid folders found, exit
#             break
#     except StopIteration:
#         break

# print(f"✅ Completed: Moved {moved_count} folders from {last_folder + 1} to {next_folder_num - 1}")





import sys
import os
import shutil
import time
from pathlib import Path
from datetime import datetime

# User inputs
source_input = input("Enter the source directory path: ").strip()
dest_input = input("Enter the destination directory path: ").strip()
last_folder = int(input("Enter the last folder number: "))

source_path = Path(source_input)
dest_path = Path(dest_input)
next_folder_num = last_folder + 1

# Check if paths are valid
if not source_path.exists() or not source_path.is_dir():
    print(f"❌ Invalid source path: {source_path}")
    sys.exit(1)

if not dest_path.exists() or not dest_path.is_dir():
    print(f"❌ Invalid destination path: {dest_path}")
    sys.exit(1)

print("🟢 Starting batch processing...\n")

moved_total = 0
last_moved_name = None
start_time = time.time()

while True:
    moved_batch = 0
    old_names = []
    new_names = []

    # ✅ Fixed parenthesis here
    folder_iter = (
        entry for entry in os.scandir(source_path)
        if entry.is_dir() and entry.name.isdigit()
    )

    for entry in folder_iter:
        new_name = str(next_folder_num)
        dest_folder = dest_path / new_name

        try:
            shutil.move(entry.path, str(dest_folder))
            current_time = datetime.now().strftime("%H:%M:%S")
            elapsed = time.time() - start_time
            print(f"[{current_time} | +{elapsed:.2f}s] Moved folder {entry.name} ➜ {new_name}")

            old_names.append(entry.name)
            new_names.append(new_name)
            last_moved_name = new_name
            next_folder_num += 1
            moved_batch += 1
            moved_total += 1
        except Exception as e:
            print(f"❌ Error moving {entry.name} to {new_name}: {e}")

        if moved_batch == 5:
            break  # Move 5 at a time

    if moved_batch == 0:
        break  # Nothing left to move

    print(f"✅ Batch Done: Moved 5 folders from {old_names[0]} to {new_names[-1]}\n")
    time.sleep(1)

print(f"\n✅ All done. Total folders moved: {moved_total}")
if last_moved_name:
    print(f"✅ Last moved folder: {last_moved_name}")
