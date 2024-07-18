import os
import cv2

import os

def list_files_in_directory(directory):
    """
    Returns a list of all filenames in the specified directory.

    Args:
        directory (str): The path to the directory.

    Returns:
        list: A list of filenames in the directory.
    """
    try:
        filenames = os.listdir(directory)
        return filenames
    except FileNotFoundError:
        print(f"The directory '{directory}' does not exist.")
        return []
    except PermissionError:
        print(f"Permission denied to access the directory '{directory}'.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

# Example usage:
# directory_path = '/path/to/directory'
# print(list_files_in_directory(directory_path))


def list_files(directory, extensions):
    """
    List all files in a directory with the specified extensions.

    Args:
    - directory (str): The directory to search in.
    - extensions (tuple): The file extensions to look for (e.g., ('.jpg', '.txt')).

    Returns:
    - list: A list of file paths that match the specified extensions.
    """
    files_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(extensions):
                files_list.append(os.path.join(root, file))
    return files_list

def read_image_and_text(directory):
    """
    Reads image files and their corresponding text files.

    Args:
    - directory (str): The directory containing the files.

    Returns:
    - list: A list of tuples, where each tuple contains an image (as a numpy array)
            and its corresponding text content (as a string).
    """
    image_text_pairs = []
    image_files = list_files(directory, ('.jpg',))
    for image_file in image_files:
        base_name = os.path.splitext(image_file)[0]
        text_file = base_name + '.txt'
        if os.path.exists(text_file):
            image = cv2.imread(image_file)
            with open(text_file, 'r') as file:
                text_content = file.read()
            image_text_pairs.append((image, text_content))
    return image_text_pairs