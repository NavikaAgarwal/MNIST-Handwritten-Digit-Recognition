import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Load dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Check shapes of the dataset
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# Plot sample images
def plot_input_img(i):
    plt.imshow(X_train[i], cmap='binary')
    plt.title(y_train[i])
    plt.show()

for i in range(10):
    plot_input_img(i)

# Normalize data
X_train = X_train.astype(np.float32) / 255
X_test = X_test.astype(np.float32) / 255

# Expand dimensions
X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)

# One-hot encode labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Define the model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPool2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPool2D((2, 2)))
model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(10, activation='softmax'))

# Display model summary
model.summary()

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Save model in the recommended Keras format
model.save("bestmodel.keras")

# Define callbacks
es = EarlyStopping(monitor='val_accuracy', min_delta=0.01, patience=4, verbose=1)
mc = ModelCheckpoint("bestmodel.keras", monitor="val_accuracy", verbose=1, save_best_only=True)
cb = [es, mc]

# Train the model
history = model.fit(X_train, y_train, epochs=50, validation_split=0.3, callbacks=cb)

# Load the best model
model_S = keras.models.load_model("bestmodel.keras")

# Evaluate the model
score = model_S.evaluate(X_test, y_test)
print(f"The model accuracy is {score[1]}")
