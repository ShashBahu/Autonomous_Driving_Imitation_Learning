# This file handles the neural network of PilotNet
# One key difference from the original paper is that we have 3 output neurons (throttle, brake & steering)

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from utils.piloterror import PilotError
import datetime
import gc
import os
import random
import matplotlib.pyplot as plt

class PilotNet():
    def __init__(self, width, height, predict=False):
        self.image_height = height
        self.image_width = width
        self.model = self.build_model() if predict == False else []
    
    def build_model(self):
        inputs = keras.Input(name='input_shape', shape=(self.image_height, self.image_width, 3))

        #x = layers.Lambda(lambda x: x/255)(inputs) #Normalisation
        l2_lambda = 1e-3
        # convolutional feature maps
        x = layers.Conv2D(filters=24, kernel_size=(5,5), strides=(2,2), activation='relu', kernel_regularizer=regularizers.l2(l2_lambda))(inputs)
        x = layers.Conv2D(filters=36, kernel_size=(5,5), strides=(2,2), activation='relu', kernel_regularizer=regularizers.l2(l2_lambda))(x)
        x = layers.Conv2D(filters=48, kernel_size=(5,5), strides=(2,2), activation='relu', kernel_regularizer=regularizers.l2(l2_lambda))(x)
        x = layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu', kernel_regularizer=regularizers.l2(l2_lambda))(x)
        x = layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu', kernel_regularizer=regularizers.l2(l2_lambda))(x)

        # flatten layer
        x = layers.Flatten()(x)

        # fully connected layers with dropouts for overfit protection
        x = layers.Dense(units=1152, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda))(x)
        x = layers.Dropout(rate=0.1)(x)
        x = layers.Dense(units=100, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda))(x)
        x = layers.Dropout(rate=0.1)(x)
        x = layers.Dense(units=50, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda))(x)
        x = layers.Dropout(rate=0.1)(x)
        x = layers.Dense(units=10, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda))(x)
        x = layers.Dropout(rate=0.1)(x)

        #Differences between this code and autodeep
        #   #Dropout 0.1 here instead of 0.2
        #   #atan output instead of linear due to difference in image outputs
        #   #No normalisation in autodeep


        # derive steering angle value from single output layer by point multiplication
        #steering_angle = layers.Dense(units=1, activation='linear')(x)
        #steering_angle = layers.Lambda(lambda X: tf.multiply(tf.atan(X), 2), name='steering_angle')(steering_angle)
        steering_angle = layers.Dense(units=1, name='steering_angle', kernel_regularizer=regularizers.l2(l2_lambda))(x)

        # derive throttle pressure value from single output layer by point multiplication
        #throttle_press = layers.Dense(units=1, activation='sigmoid', name='throttle_press')(x)

        # derive brake pressure value from single output by point multiplication
        #brake_pressure = layers.Dense(units=1, activation='sigmoid', name='brake_pressure')(x)

        # build and compile model
        #model = keras.Model(inputs = [inputs], outputs = [steering_angle, throttle_press, brake_pressure])
        model = keras.Model(inputs = [inputs], outputs = [steering_angle])

        # model.compile(
        #     optimizer = keras.optimizers.Adam(lr = 1e-4),
        #     loss = {'steering_angle': 'mse', 'throttle_press': 'mse', 'brake_pressure': 'mse'}
        # )
        model.compile(
            optimizer = keras.optimizers.Adam(lr = 1e-4),
            loss = {'steering_angle': 'mse'}
        )

        model.summary()
        return model
          

    def data_generator_sequential(self, data, batch_size):
        while True:
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                X = np.array([frame.image for frame in batch])
                #y = np.array([[frame.steering, frame.throttle, frame.brake] for frame in batch])
                y = np.array([[frame.steering] for frame in batch])
                #yield X, [y[:, 0], y[:, 1], y[:, 2]]
                yield X, [y[:, 0]]
                gc.collect()

    def train(self, name: 'Filename for saving model', data: 'Training data as an instance of pilotnet.src.Data()', epochs: 'Number of epochs to run' = 30, steps: 'Number of steps per epoch' = 10, steps_val: 'Number of steps to validate' = 10, batch_size: 'Batch size to be used for training' = 64):
        # x_train & y_train are np.array() objects with data extracted directly from the PilotData object instances

        model_path = f"models/{name}.h5"
        # Load existing model if required
        if os.path.exists(model_path):
            print(f"Loading existing model weights from {model_path}")
            self.model.load_weights(model_path)

        # fit data to model for training
        full_data = data.training_data()
        random.shuffle(full_data) 
        split_idx = int(0.8 * len(full_data))
        train_data = full_data[:split_idx]
        val_data = full_data[split_idx:]

        train_gen = self.data_generator_sequential(train_data, batch_size)
        val_gen   = self.data_generator_sequential(val_data, batch_size)
        
        self.model.fit(
            train_gen,
            steps_per_epoch=steps,       # user-defined
            epochs=epochs,
            validation_data=val_gen,
            validation_steps=steps_val,   # user-defined
            max_queue_size=2
        )

        #self.model.fit(np.array([frame.image for frame in data.training_data()]), np.array([(frame.steering, frame.throttle, frame.brake) for frame in data.training_data()]), batch_size=batch_size, epochs=epochs, steps_per_epoch=steps, validation_split=0.2, validation_steps=steps_val)
        # test the model by fitting the test data
        #stats = self.model.evaluate(np.array([frame.image for frame in data.testing_data()]), np.array([(frame.steering, frame.throttle, frame.brake) for frame in data.testing_data()]), verbose=2)
        stats = self.model.evaluate(np.array([frame.image for frame in data.testing_data()]), np.array([(frame.steering) for frame in data.testing_data()]), verbose=2)
        # print the stats

        #print(f'\nSteering loss: {stats[1]}\nModel loss: {stats[0]}\n')
        print(f'\nModel loss: {stats}\n')
        #input('\nPress [ENTER] to continue...')
        # save the trained model
        self.model.save(f"models/{name}.h5")
    
    # this method can be used for enabling the feature mentioned in app.py but needs more work
    def predict(self, data, given_model = 'default'):
        if given_model != 'default':
            try:
                # load the model
                model = keras.models.load_model(f'models/{given_model}', custom_objects = {"tf": tf})
            except Exception as e:
                raise PilotError(str(e))
                #raise PilotError('An unexpected error occured when loading the saved model. Please rerun...')
        else: model = self.model
        # predict using the model
        predictions = model.predict(data.image)
        print(predictions)
        return predictions
        