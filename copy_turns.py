import os
import json
import shutil

def copy_turning_images(folder_path, threshold=0.15, copies_per_image=1):
    copied = 0
    with os.scandir(f'{folder_path}/recordings/') as recordings:
        for recording in recordings:
            for filename in os.scandir(recording):
                try:
                    data = json.loads(filename.name[:-4])  # Parse filename
                    steering = float(data[1])

                    if steering > threshold:
                        for i in range(1, copies_per_image + 1):
                            new_data = data.copy()
                            new_data[2] += i * 1e-7  # Slightly alter throttle
                            new_name = f"{new_data}.png"
                            shutil.copy(
                                os.path.join(recording.path, filename.name),
                                os.path.join(recording.path, new_name)
                            )
                            copied += 1
                except Exception as e:
                    print(f"Error processing {filename.name}: {e}")
    print(f"Copied {copied} turning images with small throttle shift.")

# Example usage
copy_turning_images("/home/krg/Capstone/PilotNet_Train/pilotnet")
