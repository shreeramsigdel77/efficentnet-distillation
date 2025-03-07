import os

def create_directory(directory_path):
    """
    Creates a directory at the specified path if it doesn't already exist.
    If the directory already exists, it prints a message stating that.
    
    Args:
        directory_path (str): The path where the directory should be created.
        
    Returns:
        str: The path of the directory that was created or already exists.
    """
    # Create the directory if it doesn't exist
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")

    return directory_path




def create_unique_directory(base_path):
    """
    Creates a unique directory by appending a numeric suffix to the base folder name.
    If a directory with the same name already exists, it increments the counter until a unique name is found.

    Args:
        base_path (str): The base path for the directory name.

    Returns:
        str: The path of the newly created unique directory.
    """
    # Initialize a counter to append to the folder name
    counter = 1
    new_folder = base_path
    
    # Loop until we find a unique folder name
    while os.path.exists(new_folder):
        new_folder = f"{base_path}_{counter}"  # Add suffix with counter
        counter += 1
    
    # Create the new folder
    os.makedirs(new_folder)
    print(f"Project Directory Path. '{new_folder}'")

    return new_folder

