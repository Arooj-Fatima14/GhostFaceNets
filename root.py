






# import os
# from pathlib import Path
# from tqdm import tqdm

# def get_folders(path):
#     """Get list of folder names in the given path."""
#     return [f.name for f in Path(path).iterdir() if f.is_dir()]

# def rename_subfolders_sequentially(dataset_paths):
#     """Rename all subfolders across provided datasets with unique sequential numbers suffixed by dataset name."""
#     if not dataset_paths:
#         print("⚠️ No dataset paths provided.")
#         return

#     # Validate dataset paths and extract dataset names
#     datasets = []
#     for path in dataset_paths:
#         dataset_path = Path(path)
#         if not dataset_path.exists() or not dataset_path.is_dir():
#             print(f"❌ Error: Dataset path {path} does not exist or is not a directory.")
#             continue
#         dataset_name = dataset_path.name  # Extract dataset name from path
#         datasets.append((dataset_name, dataset_path))
    
#     if not datasets:
#         print("⚠️ No valid dataset folders provided.")
#         return

#     print(f"📂 Processing {len(datasets)} dataset folders: {[name for name, _ in datasets]}")
    
#     current_index = 0  # Starting index for renaming
#     total_subfolders = 0  # Track total subfolders processed
#     dataset_counts = {}  # Track count of subfolders per dataset

#     # Process each dataset folder
#     for dataset_name, dataset_path in datasets:
#         subfolders = sorted(get_folders(dataset_path))  # Get subfolders (ids) in dataset
#         dataset_counts[dataset_name] = len(subfolders)  # Store count for this dataset
#         total_subfolders += len(subfolders)
        
#         print(f"\n📁 Processing dataset: {dataset_name} ({len(subfolders)} subfolders) at {dataset_path}")

#         # Step 1: Rename subfolders to temporary names
#         temp_names = []
#         print("🚧 Step 1: Renaming to temporary names...")
#         for i, subfolder in enumerate(tqdm(subfolders, desc=f"Temp rename ({dataset_name})")):
#             temp_name = f"__tmp_{i:06d}__"
#             try:
#                 os.rename(dataset_path / subfolder, dataset_path / temp_name)
#                 temp_names.append(temp_name)
#             except Exception as e:
#                 print(f"❌ Error renaming {subfolder} → {temp_name} in {dataset_name}: {e}")

#         # Step 2: Rename to final names with dataset suffix
#         print(f"🚀 Step 2: Renaming to final names starting from {current_index}...")
#         for temp_name in tqdm(temp_names, desc=f"Final rename ({dataset_name})"):
#             final_name = f"{current_index}_{dataset_name}"
#             try:
#                 os.rename(dataset_path / temp_name, dataset_path / final_name)
#                 current_index += 1  # Increment index for next subfolder
#             except Exception as e:
#                 print(f"❌ Error renaming {temp_name} → {final_name} in {dataset_name}: {e}")

#         print(f"✅ Completed renaming for dataset: {dataset_name}. Renamed {dataset_counts[dataset_name]} subfolders.")

#     # Print summary
#     print(f"\n🎉 All datasets processed. Total subfolders renamed: {total_subfolders}")
#     print("📊 Subfolder counts per dataset:")
#     for dataset, count in dataset_counts.items():
#         print(f"  - {dataset}: {count} subfolders")
#     print(f"🔢 Last index used: {current_index - 1}")

# if __name__ == "__main__":
#     print("📥 Enter the full paths to the dataset folders (one per line). Press Enter twice to finish:")
#     dataset_paths = []
#     while True:
#         path = input().strip()
#         if path == "":
#             break
#         dataset_paths.append(path)
#     rename_subfolders_sequentially(dataset_paths)












import os
from pathlib import Path
from tqdm import tqdm

def get_folders(path):
    """Get list of folder names in the given path."""
    return [f.name for f in Path(path).iterdir() if f.is_dir()]

def rename_subfolders_sequentially(dataset_paths):
    """Rename all subfolders across provided datasets with unique sequential numbers."""
    if not dataset_paths:
        print("⚠️ No dataset paths provided.")
        return

    # Validate dataset paths and extract dataset names
    datasets = []
    for path in dataset_paths:
        dataset_path = Path(path)
        if not dataset_path.exists() or not dataset_path.is_dir():
            print(f"❌ Error: Dataset path {path} does not exist or is not a directory.")
            continue
        dataset_name = dataset_path.name  # Extract dataset name from path
        datasets.append((dataset_name, dataset_path))
    
    if not datasets:
        print("⚠️ No valid dataset folders provided.")
        return

    print(f"📂 Processing {len(datasets)} dataset folders: {[name for name, _ in datasets]}")
    
    current_index = 0  # Starting index for renaming
    total_subfolders = 0  # Track total subfolders processed
    dataset_counts = {}  # Track count of subfolders per dataset

    # Process each dataset folder
    for dataset_name, dataset_path in datasets:
        print(f">>>> Picking data from path: {dataset_path}")  # Added print statement
        subfolders = sorted(get_folders(dataset_path))  # Get subfolders (ids) in dataset
        dataset_counts[dataset_name] = len(subfolders)  # Store count for this dataset
        total_subfolders += len(subfolders)
        
        print(f">>>> Number of folders/classes in path {dataset_path}: {len(subfolders)}")  # Added print statement
        print(f"\n📁 Processing dataset: {dataset_name} ({len(subfolders)} subfolders) at {dataset_path}")

        # Step 1: Rename subfolders to temporary names
        temp_names = []
        print("🚧 Step 1: Renaming to temporary names...")
        for i, subfolder in enumerate(tqdm(subfolders, desc=f"Temp rename ({dataset_name})")):
            temp_name = f"__tmp_{i:06d}__"
            try:
                os.rename(dataset_path / subfolder, dataset_path / temp_name)
                temp_names.append(temp_name)
            except Exception as e:
                print(f"❌ Error renaming {subfolder} → {temp_name} in {dataset_name}: {e}")

        # Step 2: Rename to final names with sequential numbers only
        print(f"🚀 Step 2: Renaming to final names starting from {current_index}...")
        for temp_name in tqdm(temp_names, desc=f"Final rename ({dataset_name})"):
            final_name = f"{current_index}"  # Use only the index, no dataset name
            try:
                os.rename(dataset_path / temp_name, dataset_path / final_name)
                current_index += 1  # Increment index for next subfolder
            except Exception as e:
                print(f"❌ Error renaming {temp_name} → {final_name} in {dataset_name}: {e}")

        print(f"✅ Completed renaming for dataset: {dataset_name}. Renamed {dataset_counts[dataset_name]} subfolders.")

    # Print summary
    print(f">>>> Total number of folders/classes across all paths: {total_subfolders}")  # Added print statement
    print(f"\n🎉 All datasets processed. Total subfolders renamed: {total_subfolders}")
    print("📊 Subfolder counts per dataset:")
    for dataset, count in dataset_counts.items():
        print(f"  - {dataset}: {count} subfolders")
    print(f"🔢 Last index used: {current_index - 1}")

if __name__ == "__main__":
    print("📥 Enter the full paths to the dataset folders (one per line). Press Enter twice to finish:")
    dataset_paths = []
    while True:
        path = input().strip()
        if path == "":
            break
        dataset_paths.append(path)
    rename_subfolders_sequentially(dataset_paths)