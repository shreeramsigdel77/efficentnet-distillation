import yaml


def load_yaml(file_path):
    """Load a YAML file and return its contents as a dictionary.

    Args:
        file_path (str): The path to the YAML file to be loaded.

    Returns:
        dict or None: Returns a dictionary containing the YAML contents, 
                      or None if there was an error loading the file.
    """
    try:
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
            return data
    except yaml.YAMLError as exc:
        print(f"Error loading YAML file: {exc}")
        return None
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None