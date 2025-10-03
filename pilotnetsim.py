import carla
import numpy as np
import tensorflow as tf
import cv2
import time

import os

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

# from ctypes import *
# lib8 = cdll.LoadLibrary('/home/krg/miniconda3/envs/carla9.13_py3.7/lib/libcublas.so.11')
# lib1 = cdll.LoadLibrary('/home/krg/miniconda3/envs/carla9.13_py3.7/lib/libcudart.so.11.0')
# lib2 = cdll.LoadLibrary('/home/krg/miniconda3/envs/carla9.13_py3.7/lib/libcublasLt.so.11')
# lib3 = cdll.LoadLibrary('/home/krg/miniconda3/envs/carla9.13_py3.7/lib/libcufft.so.10')
# lib4 = cdll.LoadLibrary('/home/krg/miniconda3/envs/carla9.13_py3.7/lib/libcurand.so.10')
# lib5 = cdll.LoadLibrary('/home/krg/miniconda3/envs/carla9.13_py3.7/lib/libcusolver.so')
# lib6 = cdll.LoadLibrary('/home/krg/miniconda3/envs/carla9.13_py3.7/lib/libcusparse.so.11')
#lib7 = cdll.LoadLibrary('/usr/lib/x86_64-linux-gnu/libcudnn.so')

# import tensorrt as tr

# from tensorflow.python.compiler.tensorrt import trt_convert as trt
# print("Shou!")
# params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(precision_mode="FP32")
# print("YYDS")
# converter = trt.TrtGraphConverterV2(input_saved_model_dir="./pilotnetopt", conversion_params=params)
# converter.convert()
# converter.save("./pilotnetopt_onxx")
# print("Saved")

print("Is GPU available:", tf.config.list_physical_devices('GPU'))

i=0

def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    #client.reload_world()
    blueprint_library = world.get_blueprint_library()

    # Spawn the vehicle
    #bp = blueprint_library.filter("model3")[0]
    bp = blueprint_library.find('vehicle.dodge.charger_police') 
    spawn_point = world.get_map().get_spawn_points()[149]
    vehicle = world.spawn_actor(bp, spawn_point)
    # Set spectator to follow the vehicle
    # spectator = world.get_spectator()
    # Attach a front-facing camera
    camera_bp = blueprint_library.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", "720")
    camera_bp.set_attribute("image_size_y", "405")
    camera_bp.set_attribute("fov", "110")
    camera_init_trans = carla.Transform(carla.Location(x=0.8, z=1.7))
    camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)

    # Load trained PilotNet model
    #model = tf.keras.models.load_model("./pilotnetopt")
    print("GPU: ", tf.test.is_gpu_available)
    
    with tf.device("gpu:0"):
        #model = tf.saved_model.load("./pilotnetopt")
        model = tf.keras.models.load_model("models/PilotNet_v22.h5")
    
    print("Done loading those balls......")

    def preprocess(image):

        img = np.frombuffer(image.raw_data, dtype=np.uint8)
        print("Shape-1: ",img.shape)
        
        img = img.reshape((image.height, image.width, 4))[:, :, :3]
        print("Shape-2", img.shape)

        h, w, _ = img.shape
        top = int(0.5 * h)
        bottom = int(0.8 * h)
        left = int(0.3*w)
        right = int(0.7*w)
        img = img[top:bottom, left:right, :] 
        img = cv2.resize(img, (200, 66))    
        print("Shape-3", img.shape)
        
        #img = img.astype(np.float32) / 255.0 #- 0.5
        return np.expand_dims(img, axis=0), img

    def drive(image):
        global i
        img, img_display = preprocess(image)
        print("Output Values: ", model.predict(img))
        #print("Output Shape: ", model.predict(img)[0].shape)
        #print(model.predict(img).shape)
        
        steering = float(model.predict(img)[0].item())
        #steering = (steering * 2.0) - 1.0
        # throttling = float(model.predict(img)[1].item())
        # braking = float(model.predict(img)[2].item())
        #braking = 0 

        print("Steering: ", steering)
        # print("Throttling: ", throttling)
        # print("Braking: ", braking)
        
        control = carla.VehicleControl()
        control.steer = steering    
        # control.brake = braking
        # control.throttle = throttling
        # control.steer = np.clip(steering, -1.0, 1.0)
        
        # if braking > 0.1:   
        #     control.throttle = 0.0
        #     #control.brake = np.clip(braking, 0.0, 1.0)
        # elif steering < 0.04 and steering > -0.04:
        #     control.steer = 0
        #     #control.throttle = np.clip(throttling, 0.0, 1.0)
        #     control.brake = 0.0
        # else:
        #     control.brake = 0.0

        if steering < 0.04 and steering > -0.04:
            control.steer = 0
        
        control.throttle = 0.2
        
        #print(control.steer, control.throttle, control.brake, sep = '\n')
        print(control.steer, control.throttle, sep = '\n')
        vehicle.apply_control(control)

        i=i+1
        cv2.imwrite(f"sim_images/{i}_{steering}.png",img_display*255)
        #cv2.imshow("Image Seen: ", img_display)
        #cv2.waitKey(1)

    camera.listen(lambda image: drive(image))

    try:
        while True:

            transform = vehicle.get_transform()
            # spectator.set_transform(carla.Transform(
            #     transform.location + carla.Location(x=-10, z=2),
            #     carla.Rotation(pitch=-15, yaw=transform.rotation.yaw)
            # ))

            time.sleep(0.1)
    except KeyboardInterrupt:
        camera.stop()
        vehicle.destroy()
        print("Simulation ended. Vehicle and camera destroyed.")

if __name__ == '__main__':
    main()
