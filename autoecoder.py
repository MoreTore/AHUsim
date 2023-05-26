""" Using an autoencoder for anomaly detection generally follows a few steps:

1. Preprocessing the data: It usually involves normalization or standardization. This depends on your data. If you have already preprocessed your data, you can skip this step.

2. Splitting the data: This is generally done in a train/test split manner. We train our autoencoder on normal data.

3. Training the autoencoder: We fit the model to our training data.

4. Determining a threshold for anomaly: After training, we use our autoencoder to reconstruct the input data and calculate the reconstruction error. We then set a threshold which determines whether a data point is an anomaly or not based on the reconstruction error.

5. Detecting anomaly: We check the test data to see the anomaly. """

from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load your dataframe
df = pd.read_pickle('prepared_data.pkl')

# Check for NaN or missing values
print("Number of NaN/missing values:")
print(df.isnull().sum())
df = df.iloc[1:]
print(df)


# Normalization
scaler = MinMaxScaler()
df_normalized = scaler.fit_transform(df.values)

# Train/test split
X_train, X_test = train_test_split(df_normalized, test_size=0.2, random_state=42)

# Define the autoencoder
model = Sequential()
model.add(Dense(14, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(7, activation='relu'))
model.add(Dense(14, activation='relu'))
model.add(Dense(X_train.shape[1]))  # Multiple outputs, one for each feature

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the autoencoder
history = model.fit(X_train, X_train, epochs=3, batch_size=32, validation_split=0.2)
model.save('my_model.h5')
# Use the autoencoder for anomaly detection
X_test_predictions = model.predict(X_test)
mse = np.mean(np.power(X_test - X_test_predictions, 2), axis=1)

# Determine a suitable threshold
threshold = np.quantile(mse, 0.99)

# Detect anomalies in new data
new_data_predictions = model.predict(df_normalized)
mse_new = np.mean(np.power(df_normalized - new_data_predictions, 2), axis=1)
anomalies = mse_new > threshold


# Plot the training loss graph
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

