import os
import shutil
import re

def extract_prefix(filename):
    """
    Extracts the first string before the first underscore (_) from the filename.
    Handles Japanese characters, alphabets, and symbols properly.
    """
    match = re.match(r"([^_]+)", filename)  # Match everything before the first '_'
    return match.group(0) if match else None  # Return extracted prefix or None

def organize_images_by_prefix(source_folder):
    # Ensure the source folder exists
    if not os.path.exists(source_folder):
        print(f"Error: Source folder '{source_folder}' does not exist.")
        return
    
    # Get all files in the source directory
    files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]

    for file in files:
        prefix = extract_prefix(file)  # Extract first word/character before "_"
        
        if prefix:
            # Define the target folder path
            target_folder = os.path.join(source_folder, prefix)

            # Create the folder if it doesn't exist
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)

            # Move the file into the corresponding folder
            source_path = os.path.join(source_folder, file)
            destination_path = os.path.join(target_folder, file)
            shutil.move(source_path, destination_path)

            print(f"Moved '{file}' to '{target_folder}'")

# Example usage
# source_directory = "/home/shreeram/workspace/ambl/custom_efficent_autodistillation/Dataset/dataset_2_kana_update2/train_aug_bitwise"  # Replace with the actual folder path
source_directory = "/home/shreeram/workspace/ambl/custom_efficent_autodistillation/Dataset/dataset_2_kana_update2/test_real"
source_directory = "/home/shreeram/workspace/ambl/custom_efficent_autodistillation/Dataset/dataset_2_kana_update2/test"
source_directory = "/home/shreeram/workspace/ambl/custom_efficent_autodistillation/Dataset/dataset_2_kana_update2/valid"


source_directory = "/home/shreeram/workspace/ambl/custom_efficent_autodistillation/Dataset/dataset_0_region_update2/train_aug_bitwise"
source_directory = "/home/shreeram/workspace/ambl/custom_efficent_autodistillation/Dataset/dataset_0_region_update2/valid"
source_directory = "/home/shreeram/workspace/ambl/custom_efficent_autodistillation/Dataset/dataset_0_region_update2/test"
source_directory = "/home/shreeram/workspace/ambl/custom_efficent_autodistillation/Dataset/dataset_0_region_update2/test_real"


organize_images_by_prefix(source_directory)
