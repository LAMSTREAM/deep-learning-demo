import numpy as np               # For numerical operations
import matplotlib.pyplot as plt  # For plotting graphs

from PIL import Image            # For image processing
import os                        # For interacting with the operating system
import pandas as pd              # For data handling and manipulation

from sklearn.preprocessing import MinMaxScaler             # For normalizing data
from sklearn.model_selection import train_test_split       # For splitting the dataset
from tensorflow.keras.utils import to_categorical          # For one-hot encoding labels
from tensorflow.keras.models import Sequential             # For building a sequential neural network model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout  # Layers used in CNN
from tensorflow.keras.optimizers import Adam               # For setting the adam learning rate

from sklearn.metrics import confusion_matrix               # For confusion matrix
from sklearn.metrics import accuracy_score                 # For accuracy metric
from sklearn.metrics import precision_score                # For precision metric
from sklearn.metrics import recall_score                   # For recall metric
from sklearn.metrics import f1_score                       # For F1-score metric


data = []             # To store image data as arrays
labels = []           # To store image labels (classes)
classes = 43          # Number of unique classes (traffic sign types)
cur_path = os.getcwd()  # Get current directory path
data_folder = 'data'  # Folder where the data is stored

# adjustable parameters
learning_rate = 0.001
epochs = 15
batch_size = 32

# Loading training dataset
for i in range(classes):  # Loop over each class (0 to 42)
    path = os.path.join(cur_path, data_folder, 'Train', str(i))  # Path to the class folder
    print(path)  # Prints the path to verify it's correct
    images = os.listdir(path)  # Lists all image files in the current class folder

    for a in images:
        try:
            image = Image.open(os.path.join(path, a))  # Opens the image
            image = image.resize((30,30))       # Resizes image to 30x30 pixels
            image = np.array(image)             # Converts image to numpy array
            data.append(image)                  # Adds image data to 'data' list
            labels.append(i)                    # Adds the class label to 'labels' list
        except:
            print("Error loading image")        # Handles any errors in loading an image

# Converting lists into numpy arrays
data = np.array(data)  # Converts list of images to a numpy array
labels = np.array(labels)  # Converts list of labels to a numpy array

# Normalizing data via Min-Max normalizer
scaler = MinMaxScaler()        # Initializes a MinMaxScaler for normalization
ascolumns = data.reshape(-1, 3)  # Reshapes data into a 2D array (flattened for scaling)
t = scaler.fit_transform(ascolumns)  # Fits and scales the data to range [0, 1]
data = t.reshape(data.shape)    # Reshapes data back to its original shape (3D)
print(data.shape, labels.shape)  # Prints the shapes for verification

# Splitting training and validation dataset
X_train, X_val, y_train, y_val = train_test_split(data,
    labels, test_size=0.2, random_state=42)  # Splits data into 80% training and 20% validation
print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)  # Prints shapes for verification

# Converting the labels into one hot encoding
y_train = to_categorical(y_train, 43)  # Converts training labels to one-hot encoding
y_val = to_categorical(y_val, 43)      # Converts validation labels to one-hot encoding

# Building the model
model = Sequential()  # Initializes a sequential model
# First Convolutional Layer
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))  # 32 filters, 5x5 kernel, relu activation
model.add(MaxPool2D(pool_size=(2, 2)))  # Max pooling with 2x2 pool size
model.add(Dropout(rate=0.25))           # Dropout for regularization
# Second Convolutional Layer
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))  # 64 filters, 3x3 kernel, relu activation
model.add(MaxPool2D(pool_size=(2, 2)))  # Max pooling with 2x2 pool size
model.add(Dropout(rate=0.25))           # Dropout for regularization
# Fully Connected Layers
model.add(Flatten())                    # Flatten the 2D output to 1D
model.add(Dense(256, activation='relu')) # Dense layer with 256 neurons and relu activation
model.add(Dropout(rate=0.5))             # Dropout for regularization
model.add(Dense(43, activation='softmax')) # Output layer with 43 neurons for classification
model.summary()  # Prints a summary of the model architecture

# Compilation of the model
model.compile(loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=learning_rate), metrics=['accuracy'])  # Specifies loss, optimizer, and evaluation metric

# Training the Model
history = model.fit(X_train, y_train,
    batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val))  # Trains the model
model.save('traffic_classifier.h5')  # Saves the model to a file

# Plotting graphs for accuracy
plt.figure(0)  # Creates a new figure
plt.plot(history.history['accuracy'], label='training accuracy')  # Plots training accuracy
plt.plot(history.history['val_accuracy'], label='val accuracy')   # Plots validation accuracy
plt.title('Accuracy')  # Adds title
plt.xlabel('epochs')   # Adds x-axis label
plt.ylabel('accuracy') # Adds y-axis label
plt.legend()           # Adds legend
plt.show()             # Displays plot

plt.figure(1)  # Creates another figure
plt.plot(history.history['loss'], label='training loss')  # Plots training loss
plt.plot(history.history['val_loss'], label='val loss')   # Plots validation loss
plt.title('Loss')       # Adds title
plt.xlabel('epochs')    # Adds x-axis label
plt.ylabel('loss')      # Adds y-axis label
plt.legend()            # Adds legend
plt.show()              # Displays plot

# Testing the model
path = os.path.join(cur_path, data_folder)  # Sets the path to the test folder
y_test = pd.read_csv(os.path.join(data_folder, 'Test.csv'))  # Reads test labels from CSV file
labels = y_test["ClassId"].values  # Extracts class IDs as true labels
imgs = y_test["Path"].values       # Extracts file paths of test images

data=[]  # Resets data list for test images

for img in imgs:
    image = Image.open(os.path.join(path, img))  # Opens test image
    image = image.resize((30,30))         # Resizes to 30x30 pixels
    data.append(np.array(image))          # Adds image data to list

X_test = np.array(data)  # Converts list to numpy array

# Normalizing test set
ascolumns = X_test.reshape(-1, 3)  # Reshapes for normalization
t = scaler.transform(ascolumns)    # Scales test data based on training scaler
X_test = t.reshape(X_test.shape)   # Reshapes back to 3D

# Making Predicting on test set
pred = np.argmax(model.predict(X_test), axis=1)  # Predicts and retrieves the class with highest probability

# Evaluating Model Performance
# accuracy: (tp + tn) / (p + n)
# Confusion Matrix
cm = confusion_matrix(labels, pred)  # Generates confusion matrix
print('Confusion Matrix:')           # Prints header
print(cm)                            # Prints matrix
# Accuracy
accuracy = accuracy_score(labels, pred)
print('Accuracy: %f' % accuracy)
# Precision
precision = precision_score(labels, pred, average='macro')
print('Precision: %f' % precision)
# Recall
recall = recall_score(labels, pred, average='macro')
print('Recall: %f' % recall)
# F1 Score
f1 = f1_score(labels, pred, average='macro')
print('F1 score: %f' % f1)


