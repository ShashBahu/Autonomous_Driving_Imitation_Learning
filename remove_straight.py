import os
import ast
import random
import json

r=0.5
def remove_low_steering_images(folder_path, threshold=0.0075):
    removed = 0
    with os.scandir(f'{folder_path}/recordings/') as recordings:
        for recording in recordings:
            for filename in os.scandir(recording):
                try:
                    # Extract the values from the filename
                    data = json.loads(filename.name[:-4])
                    steering = float(data[1])
                    if abs(steering-0.2) <= threshold and r>random.random():
                    #if steering >= threshold and steering <= threshold and r>random.random():
                        os.remove(os.path.join(folder_path, filename))
                        print("Removed: ",filename.name)
                        removed += 1
                except Exception as e:
                        print(f"Error processing {filename.name}: {e}")
            print(f"Removed {removed} images with near-zero steering.")

# Example usage
remove_low_steering_images("/home/krg/Capstone/PilotNet_Train/pilotnet")