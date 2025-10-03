import os
import shutil

# Source directory
image_dir = "/home/krg/Capstone/PilotNet_Train/pilotnet/recordings/2025-06-03@15:13:31"

# Destination directory for sorted files
output_dir = "/home/krg/Capstone/PilotNet_Train/pilotnet/sorted_images"
os.makedirs(output_dir, exist_ok=True)

# Filter out non-image files and get full paths
image_files = [f for f in os.listdir(image_dir)]

# Sort by file modification time
sorted_images = sorted(image_files, key=lambda f: os.path.getmtime(os.path.join(image_dir, f)))

# Now sorted_images contains filenames in the order they were written to disk

# Copy to new directory with indexed filenames
for i, filename in enumerate(sorted_images):
    src_path = os.path.join(image_dir, filename)
    dst_path = os.path.join(output_dir, f"sorted_image_{i}.png")  # zero-padded filenames like image_00001.png
    shutil.copy2(src_path, dst_path)  # preserves metadata (timestamps, etc.)
    print(f"Copied {filename} -> {dst_path}")