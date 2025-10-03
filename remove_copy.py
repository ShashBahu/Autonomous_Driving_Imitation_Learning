import os

def remove_copy0_images(folder_path):
    removed = 0
    with os.scandir(f'{folder_path}/recordings/') as recordings:
        for recording in recordings:
            for file in os.scandir(recording):
                if "_copy0" in file.name and file.name.endswith(".png"):
                    try:
                        os.remove(file.path)
                        print(f"Removed: {file.name}")
                        removed += 1
                    except Exception as e:
                        print(f"Error removing {file.name}: {e}")
    print(f"\nTotal removed: {removed} images containing '_copy0'.")

# Example usage
remove_copy0_images("/home/krg/Capstone/PilotNet_Train/pilotnet")
