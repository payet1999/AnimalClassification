# Author: Adrià Payet
# Date: 04/04/2023
# DDBB: https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset + shiba inu (30 pics)
# Model -> loss: 1.3236 - accuracy: 0.6195 - val_loss: 1.4665 - val_accuracy: 0.5954

# Import libraries
import argparse
import os  # For interacting with the file system
import shutil  # For managing files and directories in a cross-platform manner

import keras  # For building deep learning models
# Training callbacks
from keras.callbacks import EarlyStopping, ModelCheckpoint  # For training callbacks
# Model architecture
from keras import Sequential  # For building sequential models
# Pre-trained models
from keras.applications import InceptionV3, ResNet152V2, Xception  # For using pre-trained models
from keras.layers import Dense, Dropout, GlobalAvgPool2D as GAP  # For defining model layers
from keras.models import load_model  # For loading pre-trained models
# Data preprocessing
from keras.preprocessing.image import ImageDataGenerator  # For image data augmentation
import matplotlib.pyplot as plt  # For creating static plots
import numpy as np  # For numerical operations on arrays
import plotly.graph_objs as go  # For interactive visualizations
import scipy
# Data visualization
import seaborn as sns  # For statistical visualizations
import tensorflow.lite.python.lite
from tqdm import tqdm  # For progress bars

# from glob import glob  # For finding file paths


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train and export ResNet152V2 model for Dog Breed Classification."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="C:/Users/adria/Desktop/Projectes/DogBreedsClassification/images/Images",
        help="Path to the dataset directory"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--img_height", type=int, default=256, help="Target image height")
    parser.add_argument("--img_width", type=int, default=256, help="Target image width")
    parser.add_argument(
        "--output_model",
        type=str,
        default="dogBreedClassificationModel.tflite",
        help="Output path for the TFLite model"
    )
    return parser.parse_args()


def pie_chart(class_names, class_sizes):
    """
    Plot a pie graph of the number of images in each class
    :return:
    """
    # Define the data
    data = go.Pie(labels=class_names, values=class_sizes)

    # Define the layout
    layout = go.Layout(title={"text": "Class Distribution", "x": 0.5})

    # Create the figure
    fig = go.Figure(data=data, layout=layout)

    # Display the figure
    fig.show()



def bar_chart(class_names, class_sizes):
    """
    Plot a bar graph of the number of images in each class
    :return:
    """

    # Set the size of the figure
    plt.figure(figsize=(10, 5))

    # Plot a bar chart using the class names as the x-axis and class sizes as the y-axis
    sns.barplot(x=class_names, y=class_sizes)

    # Add a grid to the plot
    plt.grid()

    # Add a horizontal line to show the mean number of images across all classes
    plt.axhline(
        np.mean(class_sizes),
        color='black',
        linestyle=':',
        label="Average number of images per class"
    )

    # Add a legend to the plot
    plt.legend()

    # Show the plot
    plt.show()


def show_image(image, image_title=None):
    '''
    This function takes in an image and an optional title and plots the image.
    '''
    # Display the image
    plt.imshow(image)

    # Set the title of the plot if provided
    plt.title(image_title)

    # Turn off the axes in the plot
    plt.axis('off')


def get_random_data(data_tuple):
    """
    Function to get a random data point from a given dataset.

    Args:
    data_tuple (tuple): A tuple containing the dataset images and labels as numpy arrays.

    Returns:
    A random image and its corresponding label as numpy arrays.
    """
    images, labels = data_tuple
    # get a random index for an image in the dataset
    idx = np.random.randint(len(images))

    # select the image and its corresponding label using the random index
    image, label = images[idx], labels[idx]

    # return the selected image and label
    return image, label


def plot_dataset(train_data, class_names):
    # Set the figure size for the plot
    plt.figure(figsize=(20, 20))

    # Initialize a counter for the subplots
    counter = 1

    # Loop over the train dataset
    for images, labels in iter(train_data):

        # Get a random image and label
        image, label = get_random_data([images, labels])

        # Plot the image with its class name as the title
        plt.subplot(5, 5, counter)
        show_image(image, image_title=f"Class : {class_names[int(label)]}")

        # Increment the counter
        counter += 1

        # End the loop when 25 images have been plotted
        if counter >= 26:
            break

    # Adjust the layout and display the plot
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    args = parse_args()

    """ Exploring the Stanford Dogs Dataset: Obtaining Class Names and Counting Classes """
    # Set the path to the dataset
    data_path = args.data_path

    # Get a list of class names from the data path
    class_names = sorted(os.listdir(data_path))

    # Count the number of classes
    num_classes = len(class_names)

    # Print the class names and the total number of classes
    print("Class Names:")
    print(' '.join(class_names))
    print("Number of Classes:", num_classes)

    """ Examining Class Distribution in the Dataset """
    # Get the number of samples in each class
    class_sizes = []
    for name in class_names:
        class_size = len(os.listdir(os.path.join(data_path, name)))
        class_sizes.append(class_size)

    # Print the class distribution
    print("Class Distribution:\n", class_sizes)

    """ Visualizing Class Distribution in the Animal-10 Dataset using a Pie Chart and Bar Graph """
    # pie_chart(class_names, class_sizes)

    # bar_chart(class_names, class_sizes)

    """ Data Preparation and Augmentation """
    # Initialize Generator with the specified image transformations and preprocessing
    # rescale: normalizes pixel values from 0-255 to 0-1
    # horizontal_flip: randomly flips images horizontally
    # vertical_flip: randomly flips images vertically
    # rotation_range: randomly rotates images by a given range in degrees
    # validation_split: splits the data into training and validation sets, with 20% of the data used for validation
    data_generator = ImageDataGenerator(
        rescale=1. / 255,
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=20,
        validation_split=0.2
    )

    # Target resolution settings
    target_size = (args.img_height, args.img_width)
    input_shape = (args.img_height, args.img_width, 3)

    # Load training data from the specified directory and apply the generator
    # target_size: resizes the images to a specified size
    # class_mode: specifies the type of label encoding, binary for 2 classes
    # batch_size: specifies the number of samples per batch
    # shuffle: shuffles the data after each epoch
    # subset: specifies the subset of data to load, in this case, the training set
    train_data = data_generator.flow_from_directory(
        data_path,
        target_size=target_size,
        class_mode='binary',
        batch_size=args.batch_size,
        shuffle=True,
        subset='training'
    )

    # Load validation data from the specified directory and apply the generator
    # subset: specifies the subset of data to load, in this case, the validation set
    valid_data = data_generator.flow_from_directory(
        data_path,
        target_size=target_size,
        class_mode='binary',
        batch_size=args.batch_size,
        shuffle=True,
        subset='validation'
    )

    """ Data Visualization """
    # plot_dataset(train_data, class_names)

    """ Model Training """
    # Specify the name of the model as "ResNet152V2".
    name = "ResNet152V2"

    # Load the pre-trained ResNet152V2 model, freeze its weights and exclude its final classification layer.
    base_model = ResNet152V2(
        include_top=False, input_shape=input_shape, weights='imagenet'
    )
    base_model.trainable = False

    # Create a sequential model with the ResNet152V2 base model, a global average pooling layer, two fully connected layers, and a final softmax classification layer.
    resnet152V2 = Sequential([
        base_model,
        GAP(),
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ], name=name)

    # Compile the model with sparse categorical cross-entropy as the loss function, Adam optimizer and accuracy as the evaluation metric.
    resnet152V2.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    # Set up the EarlyStopping and ModelCheckpoint callbacks to monitor the training process and save the best model weights.
    cbs = [
        EarlyStopping(patience=3, restore_best_weights=True),
        ModelCheckpoint(name + ".h5", save_best_only=True)
    ]

    # Train the model using the training and validation datasets, using 50 epochs and the previously defined callbacks.
    resnet152V2.fit(
        train_data, validation_data=valid_data,
        epochs=args.epochs, callbacks=cbs
    )

    """ Download Model for later use in Android app. Optimized version to be lighter than 200MB """
    keras.models.save_model(resnet152V2, 'resnet152V2.pbtxt')
    converter = tensorflow.lite.python.lite.TFLiteConverter.from_keras_model(
        model=resnet152V2
    )

    converter.optimizations = [tensorflow.lite.python.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tensorflow.lite.python.lite.float16]

    model_tflite = converter.convert()
    with open(args.output_model, "wb") as f:
        f.write(model_tflite)