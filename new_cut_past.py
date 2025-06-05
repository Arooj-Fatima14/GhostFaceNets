# import os
# import shutil

# # Paths
# dataset_path = r"C:\Users\HP\Documents\zmine\parttime\ghostfacerepo\projectrepo\datasets\faces_umd\hybrid"  # to
# new_data_path = r"C:\Users\HP\Documents\zmine\parttime\ghostfacerepo\projectrepo\datasets\New folder\old"  # from

# # Step 1: Find the highest folder number in the dataset
# try:
#     existing_folders = [int(f) for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f)) and f.isdigit()]
#     last_folder = max(existing_folders) if existing_folders else -1  # Should be 8270 if folders are 0 to 8270
# except FileNotFoundError:
#     print(f"Error: Dataset path {dataset_path} not found.")
#     exit(1)
# except ValueError:
#     print("Error: Non-numeric folder names found in dataset path.")
#     exit(1)

# # Step 2: Set the starting number for new folders
# start_folder = last_folder + 1  # Start from 8271

# # Step 3: Get list of new data folders (assuming they are named 0, 1, 2, ...)
# try:
#     new_data_folders = [f for f in os.listdir(new_data_path) if os.path.isdir(os.path.join(new_data_path, f)) and f.isdigit()]
#     new_data_folders.sort(key=int)  # Sort to ensure 0, 1, 2, ... order
# except FileNotFoundError:
#     print(f"Error: New data path {new_data_path} not found.")
#     exit(1)

# # Step 4: Move new folders to the dataset
# moved_count = 0
# for i, folder in enumerate(new_data_folders):
#     new_folder_name = str(start_folder + i)  # New name (e.g., 8271, 8272, ...)
#     source_folder = os.path.join(new_data_path, folder)
#     destination_folder = os.path.join(dataset_path, new_folder_name)
   
#     try:
#         # Move the entire folder to the dataset with the new name
#         shutil.move(source_folder, destination_folder)
#         print(f"Moved folder {folder} to {new_folder_name}")
#         moved_count += 1
#     except shutil.Error as e:
#         print(f"Error moving folder {folder} to {new_folder_name}: {e}")
#     except OSError as e:
#         print(f"OS error while moving folder {folder} to {new_folder_name}: {e}")

# print(f"Moved {moved_count} new folders starting from {start_folder}")




import os
import shutil

# === Get user input for source and destination paths ===
dataset_path = input("Enter the full path to the destination dataset (e.g., hybrid folder): ").strip()
new_data_path = input("Enter the full path to the source of new folders: ").strip()

# Step 1: Find the highest-numbered folder in dataset_path
try:
    existing_folders = [
        int(f) for f in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, f)) and f.isdigit()
    ]
    last_folder = max(existing_folders) if existing_folders else -1
except FileNotFoundError:
    print(f"❌ Error: Dataset path '{dataset_path}' not found.")
    exit(1)
except ValueError:
    print("❌ Error: Dataset folder names must be numeric.")
    exit(1)

# Step 2: Determine start number for new folders
start_folder = last_folder + 1

# Step 3: Collect new folders to be moved
try:
    new_data_folders = [
        f for f in os.listdir(new_data_path)
        if os.path.isdir(os.path.join(new_data_path, f)) and f.isdigit()
    ]
    new_data_folders.sort(key=int)
except FileNotFoundError:
    print(f"❌ Error: New data path '{new_data_path}' not found.")
    exit(1)

# Step 4: Move and rename folders
moved_count = 0
for i, folder in enumerate(new_data_folders):
    new_folder_name = str(start_folder + i)
    source_folder = os.path.join(new_data_path, folder)
    destination_folder = os.path.join(dataset_path, new_folder_name)

    try:
        shutil.move(source_folder, destination_folder)
        print(f"✅ Moved folder '{folder}' ➜ '{new_folder_name}'")
        moved_count += 1
    except shutil.Error as e:
        print(f"⚠️ Error moving folder '{folder}' ➜ '{new_folder_name}': {e}")
    except OSError as e:
        print(f"⚠️ OS error moving folder '{folder}' ➜ '{new_folder_name}': {e}")

print(f"\n✅ Total moved folders: {moved_count}, starting from {start_folder}")
