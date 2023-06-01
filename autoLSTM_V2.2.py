import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from keras import regularizers
import numpy as np
import pandas as pd
from enum import Enum
import joblib
import os
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from generator import DataGenerator
import tensorflow_addons as tfa
import tensorflow as tf
from keras.models import load_model

# Define a function for the learning rate schedule
def scheduler(epoch, lr):
    if epoch < 2:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

# Create the learning rate scheduler callback
lr_scheduler = LearningRateScheduler(scheduler)
# Create EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

class FaultTypes(Enum):
    NONE = 0
    OUTSIDE_AIR_TEMP = 1
    RETURN_AIR_TEMP = 2
    MIXED_AIR_DAMPER_POSITION = 3
    COOLING_VALVE_POSITION = 4
    HEATING_VALVE_POSITION = 5

params = {'dim': (60,7),
          'batch_size': 32,
          'n_classes': len(FaultTypes),
          'shuffle': False}

DATA_DIR = 'data/'
list_IDs = os.listdir(DATA_DIR)

def data_generator(files, batch_size=32):
    # Loop indefinitely
    while True:
        for file in files:
            #scaler = StandardScaler()
            # Load the dataframe
            df = pd.read_pickle(os.path.join(DATA_DIR, file))
            df = df.iloc[1:] # remove the first row of data

            # Preprocess the dataframe
            df_norm = scaler.transform(df.drop(columns=['fault_type']))
            y = to_categorical(df['fault_type'], num_classes=len(FaultTypes))

            # Slice the dataframe into sequences
            X = np.array([df_norm[i:i+60] for i in range(len(df_norm)-60)])
            y = y[60:]

            # Yield batches
            for i in range(0, len(X), batch_size):
                yield X[i:i+batch_size], y[i:i+batch_size]

# Load the existing model
model = load_model('models/autoencoder_modelV2_retrained.h5')
# Load the existing scaler
scaler = joblib.load('models/scalerV2.2.pkl')
# Get a list of all pickle files in the directory
files = [f for f in os.listdir(DATA_DIR) if f.endswith('.pkl')]
# Define the split point for train and validation files
split_index = int(len(files) * 0.8)
# Create the generators
train_generator = data_generator(files[:split_index])
validation_generator = data_generator(files[split_index:])
# Calculate the steps per epoch for both training and validation
steps_per_epoch_train = len(files[:split_index]) * 100
steps_per_epoch_val = len(files[split_index:]) * 100

# Create the generator
generator = data_generator(files)

#model = Sequential()
#
#model.add(LSTM(128, input_shape=(60, 7), return_sequences=True))
#model.add(LSTM(128))
#model.add(Dense(len(FaultTypes), activation='softmax'))
#model.compile(loss='categorical_crossentropy',
#              optimizer=tfa.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
#              metrics=['accuracy'])

# Train the model on each file for one epoch
for file in files[:split_index]:
    print(f"Training on file {file}...")
    model.fit(train_generator,
              steps_per_epoch=steps_per_epoch_train,
              validation_data=validation_generator,
              validation_steps=steps_per_epoch_val,
              epochs=1,
              callbacks=[early_stopping, lr_scheduler])
    model.save('models/autoencoder_modelV2_retrained.h5')
