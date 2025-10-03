from utils.screen import message
import os, json, cv2, gc
import random, albumentations as A
import numpy as np


class PilotData(object):
    # given the path to an image file as argument, return a PilotData object
    def __init__(self, path_to: 'Path to the image file', image_file: 'An image file with driving data as filename' = '', image_width=200, image_height=66 ,isTraining = True):
        
        # Define the augmentation pipeline globally (ReplayCompose)
        # self.augmentor = A.ReplayCompose([
        #     A.RandomBrightnessContrast(p=0.5),
        #     A.HueSaturationValue(p=0.5),
        #     A.GaussianBlur(p=0.2),
        #     A.RandomRain(p=0.1),
        #     A.RandomSnow(p=0.1),
        #     A.RandomFog(p=0.1),
        #     A.RandomSunFlare(p=0.1, src_radius=50, angle_lower=0.3, angle_upper=0.7),
        #     A.RandomShadow(p=0.2),
        #     A.HorizontalFlip(p=0.5),
        # ])
        
        self.steering, self.image = self.parse_train(path_to, image_file, image_width, image_height) if isTraining == True else self.parse_test(path_to, image_width, image_height)

    # this method overrides str() for PilotData objects
    def __str__(self):
        return f'PilotData(For the given image frame, the telemetry states that steering should be at {self.steering}, brakes should be pressed {self.brake} units & throttle should be held at {self.throttle})'

    def __repr__(self):
        return f"PilotData(steering={self.steering}, throttle={self.throttle}, brake={self.brake} image={self.image})"

    
    # def augment_image(self, image, steering_angle, apply_aug_prob=0.7):
        
    #     if random.random() > apply_aug_prob:
    #         # Skip augmentation, return original
    #         return image, steering_angle

    #     augmented = self.augmentor(image=image)
    #     image_aug = augmented["image"]

    #     for tf in augmented.get("replay", {}).get("transforms", []):
    #         if tf["__class_fullname__"] == "HorizontalFlip" and tf.get("applied", False):
    #             steering_angle = -steering_angle

    #     return image_aug, steering_angle


    # create data for training
    def parse_train(self, path_to, image_file, image_width, image_height):
        # remove the last 4 characters from filename which would be the file extension
        # then parse the resulting string into a list
        data = json.loads(image_file[:-4])
        # read and resize the image
        image = cv2.imread(path_to)
        h, w, _ = image.shape
        top = int(0.5 * h)
        bottom = int(0.8 * h)
        left = int(0.3*w)
        right = int(0.7*w)
        image = image[top:bottom, left:right, :] 
        #resize = cv2.resize(cropped, (360, 120))
        image = cv2.resize(image, (image_width, image_height))
        #image, data[1] = self.augment_image(image, data[1])
        return (data[1], image)
    
    # create data for prediction
    def parse_test(self, file_path, image_width, image_height):
        # read and resize the image
        image = cv2.imread(file_path)
        # h, w, _ = image.shape
        # top = int(0.5 * h)
        # bottom = int(0.8 * h)
        # left = int(0.3*w)
        # right = int(0.7*w)
        # image = image[top:bottom, left:right, :] 
        image = cv2.resize(image, (image_width, image_height))
        # reshape image to make it consumable for the Input neuron
        print("Staryu")
        image = image.reshape(image_height, image_width, 3)
        cv2.imwrite(f"app_images/1.png",image)
        image = image.astype(np.float32) / 255.0 
        image = np.expand_dims(image, axis=0)
        print(image.shape)
        print("Golem")
        cv2.imwrite(f"app_images/2.png",image)
        print("Test parse")
        return (0, image)

class Data():
    def __init__(self, image_width, image_height, count, training_batch, isTraining: 'whether to prepare data for training or prediction' = True):
        self.data = self.generate_data(image_width, image_height, count, training_batch)

    def generate_data(self, image_width, image_height, count, training_batch):
        gc.collect()
        data = []
        all_images = []
        # get context & loop though each directory
        with os.scandir('recordings/') as recordings:
            for recording in recordings:
                # get context & loop through each image
                message(f'Extracting from {recording.name}')
                with os.scandir(recording) as it:
                    images = list(it)
                    print("Modification Time: ", images[0].stat().st_mtime)
                    all_images.extend(images)
        # Sort by file modification time
        sorted_images = sorted(all_images, key=lambda f: f.stat().st_mtime)
        print("Sort hai bro")

        start = count * training_batch
        end = (count+1) * training_batch
        
        if end > len(sorted_images):
            end = len(sorted_images)

        for image in sorted_images[start:end]:
            # add a new PilotData instance to array
            data.append(PilotData(image.path, image.name, image_width, image_height))
            if(len(data)%1000==0):
                print("Data collected:", len(data), "count =", count)
                                    
        return data

    # return the first 3/4 of total data
    def training_data(self):
        return self.data[:int((len(self.data)*3)/4)]
    
    # return last 1/4 of total data
    def testing_data(self):
        return self.data[int((len(self.data)*3)/4)+1:]