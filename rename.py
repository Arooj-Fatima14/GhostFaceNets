import os

def rename_folders_sequentially(parent_folder):
    # Get all items in the parent folder
    items = os.listdir(parent_folder)

    # Filter only directories and sort them
    folders = [item for item in items if os.path.isdir(os.path.join(parent_folder, item))]
    folders.sort()

    # Rename folders to temporary names to avoid conflicts
    temp_names = []
    for i, folder in enumerate(folders):
        temp_name = f"temp_rename_{i}"
        os.rename(os.path.join(parent_folder, folder), os.path.join(parent_folder, temp_name))
        temp_names.append(temp_name)

    # Final renaming from 0 onward
    for index, temp_name in enumerate(temp_names):
        os.rename(os.path.join(parent_folder, temp_name), os.path.join(parent_folder, str(index)))
        print(f"Renamed: {temp_name} → {index}")

if __name__ == "__main__":
    folder_path = input("Enter the path to the parent folder: ").strip()
    if os.path.isdir(folder_path):
        rename_folders_sequentially(folder_path)
    else:
        print("Invalid path. Please make sure the folder exists.")
