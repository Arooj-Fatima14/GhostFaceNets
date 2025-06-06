import os
from tqdm import tqdm

def get_folders_scandir(path):
    with os.scandir(path) as entries:
        for entry in entries:
            if entry.is_dir():
                yield entry.name

def rename_folders_sequentially(parent_folder, start_index=0):
    try:
        folders = list(get_folders_scandir(parent_folder))
    except Exception as e:
        print(f"❌ Error reading folder: {e}")
        return

    total = len(folders)
    if total == 0:
        print("⚠️ No folders found to rename.")
        return

    print(f"\n📂 Total folders found: {total}")
    print(f"🔢 Renaming will start from: {start_index}\n")

    # Step 1: Rename to temporary names
    print("🚧 Step 1: Renaming to temporary names...")
    temp_names = []
    for i, folder in enumerate(tqdm(folders, desc="Renaming to temp")):
        temp_name = f"__tmp_{i:06d}__"
        try:
            os.rename(os.path.join(parent_folder, folder), os.path.join(parent_folder, temp_name))
            temp_names.append(temp_name)
        except Exception as e:
            print(f"❌ Error renaming {folder} → temp: {e}")
    print("✅ Step 1 complete.\n")

    # Step 2: Rename to final names
    print("🚀 Step 2: Renaming to final numbers...")
    for i, temp_name in enumerate(tqdm(temp_names, desc="Renaming to final")):
        final_name = str(start_index + i)
        try:
            os.rename(os.path.join(parent_folder, temp_name), os.path.join(parent_folder, final_name))
        except Exception as e:
            print(f"❌ Error renaming {temp_name} → {final_name}: {e}")

    print("\n✅ All folders renamed successfully.")

if __name__ == "__main__":
    folder_path = input("📥 Enter the full path to the parent folder: ").strip()
    if not os.path.isdir(folder_path):
        print("❌ Invalid path. Please make sure the folder exists.")
    else:
        try:
            start_index = int(input("🔢 Enter the starting number for renaming (e.g., 0 or 1000): ").strip())
        except ValueError:
            print("❌ Invalid number. Using default start = 0")
            start_index = 0
        rename_folders_sequentially(folder_path, start_index)
