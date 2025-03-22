import os


def get_files_folder(folder_path, extension=None):
    """
    Get all files in a folder
    """
    files = []
    for file in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file)) and extension in file:
            files.append(file)

    return files
