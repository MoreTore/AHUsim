from keras.models import Sequential
from keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


# Load your dataframe
df = pd.read_pickle('prepared_data.pkl')

# Normalization
scaler = MinMaxScaler()
df_normalized = scaler.fit_transform(df.values)

# Convert your data to sequences
sequence_length = 60  # This value can be tuned
X = [df_normalized[i: i + sequence_length] for i in range(df_normalized.shape[0] - sequence_length)]
X = np.array(X)

# Train/test split
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# Define the LSTM autoencoder
model = Sequential()

# Encoder
model.add(LSTM(128, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(LSTM(64, activation='relu', return_sequences=False))

# This layer converts the 3D output to 2D
model.add(RepeatVector(X_train.shape[1]))

# Decoder
model.add(LSTM(64, activation='relu', return_sequences=True))
model.add(LSTM(128, activation='relu', return_sequences=True))

model.add(TimeDistributed(Dense(X_train.shape[2])))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the autoencoder
model.fit(X_train, X_train, epochs=3, batch_size=32, validation_split=0.1)

# Use the autoencoder for anomaly detection
X_test_predictions = model.predict(X_test)
mse = np.mean(np.power(X_test - X_test_predictions, 2), axis=(1,2))

# Determine a suitable threshold
threshold = np.quantile(mse, 0.99)

# Detect anomalies in new data
new_data_predictions = model.predict(X)
mse_new = np.mean(np.power(X - new_data_predictions, 2), axis=(1,2))
anomalies = mse_new > threshold
