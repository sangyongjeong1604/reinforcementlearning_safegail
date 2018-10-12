import numpy as np

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Conv2D


# Define model
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(75, 320, 3)))
model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="elu"))
model.add(Conv2D(36, (5, 5), strides=(2, 2), activation="elu"))
model.add(Conv2D(48, (5, 5), strides=(2, 2), activation="elu"))
model.add(Conv2D(64, (3, 3), activation="elu"))
model.add(Conv2D(64, (3, 3), activation="elu"))
model.add(Dropout(0.8))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Train model
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
model.fit(X_train, y_train, epochs=5, validation_split=0.1)

# Save model
model.save('model.h5')
