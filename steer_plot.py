import os
import json
import matplotlib.pyplot as plt

all_images = []
with os.scandir('recordings/') as recordings:
    for recording in recordings:
        # get context & loop through each image
        print(f'Extracting from {recording.name}')
        with os.scandir(recording) as it:
            images = list(it)
            print("Modification Time: ", images[0].stat().st_mtime)
            all_images.extend(images)

steering = []
for image in all_images:
    data = json.loads(image.name[:-4])
    steering.append(data[1])

plt.hist(steering, bins=100)
plt.title("Steering Angle Distribution")
plt.show()
