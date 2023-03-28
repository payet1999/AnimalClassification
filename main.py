# Author: Adrià Payet
# Date: 23/03/2023
# Credits: https://www.kaggle.com/code/divinechisomukonu/animal-species-classification-using-cnn
# Model -> loss: 0.2883 - accuracy: 0.9055 - val_loss: 0.3348 - val_accuracy: 0.8844


""" 1. Importing Libraries for Animal Species Classification """

# Import libraries
import os  # For interacting with the file system
import shutil  # For managing files and directories in a cross-platform manner
import keras  # For building deep learning models
import numpy as np  # For numerical operations on arrays
# from glob import glob  # For finding file paths
import tensorflow.lite.python.lite
from tqdm import tqdm  # For progress bars

# Data preprocessing
from keras.preprocessing.image import ImageDataGenerator  # For image data augmentation

# Data visualization
import seaborn as sns  # For statistical visualizations
import plotly.graph_objs as go  # For interactive visualizations
import matplotlib.pyplot as plt  # For creating static plots

# Model architecture
from keras import Sequential  # For building sequential models
from keras.models import load_model  # For loading pre-trained models
from keras.layers import Dense, GlobalAvgPool2D as GAP, Dropout  # For defining model layers

# Training callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping  # For training callbacks

# Pre-trained models
from keras.applications import InceptionV3, Xception, ResNet152V2  # For using pre-trained models


def pie_chart():
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


def bar_chart():
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
    plt.axhline(np.mean(class_sizes), color='black', linestyle=':', label="Average number of images per class")

    # Add a legend to the plot
    plt.legend()

    # Show the plot
    plt.show()


def create_directory():
    """
    In the code below, we create a smaller, more manageable dataset for training and testing machine learning
    models. We also mapped Italian names to English names to create more understandable naming convention for the
    classes in the dataset.
    :return:
    """
    # Create the sampled data directory if it doesn't exist
    if not os.path.exists(sampled_data_path):
        os.mkdir(sampled_data_path)

    # Set the percentage of each class to sample
    sample_percent = 0.1

    # Define a dictionary that maps the original class names to their English names
    class_names_dict = {
        'cane': 'dog',
        'cavallo': 'horse',
        'elefante': 'elephant',
        'farfalla': 'butterfly',
        'gallina': 'chicken',
        'gatto': 'cat',
        'mucca': 'cow',
        'pecora': 'sheep',
        'ragno': 'spider',
        'scoiattolo': 'squirrel'
    }

    # Loop through each class directory and copy a percentage of the images to the sampled data directory
    for class_name in os.listdir(data_path):
        # Get the path to the original class directory
        class_path = os.path.join(data_path, class_name)
        # Get the English name of the class
        class_name_en = class_names_dict[class_name]
        # Get the path to the sampled class directory
        sampled_class_path = os.path.join(sampled_data_path, class_name_en)
        # Create the sampled class directory if it doesn't exist
        if not os.path.exists(sampled_class_path):
            os.mkdir(sampled_class_path)
        # Get a list of all the image files in the class directory
        image_files = os.listdir(class_path)
        # Calculate the number of images to sample
        num_images = int(len(image_files) * sample_percent)
        # Sample the images
        sampled_images = np.random.choice(image_files, size=num_images, replace=False)
        # Copy the sampled images to the sampled class directory
        for image_name in sampled_images:
            src_path = os.path.join(class_path, image_name)
            dst_path = os.path.join(sampled_class_path, image_name)
            shutil.copyfile(src_path, dst_path)


def sampled_pie_chart():
    # Define the data
    data = go.Pie(labels=class_names, values=class_sizes)

    # Define the layout
    layout = go.Layout(title={"text": "Class Distribution", "x": 0.5})

    # Create the figure
    fig = go.Figure(data=data, layout=layout)

    # Display the figure
    fig.show()


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


def plot_dataset():
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
        if counter >= 26: break

    # Adjust the layout and display the plot
    plt.tight_layout()
    plt.show()

'''
Main Init
'''
if __name__ == '__main__':


    """ 2. Exploring the Animal-10 Dataset: Obtaining Class Names and Counting Classes """
    # Set the path to the dataset
    data_path = 'C:/Users/adria/Desktop/Projectes/AnimalClassification/raw-img'

    # Get a list of class names from the data path
    class_names = sorted(os.listdir(data_path))

    # Count the number of classes
    num_classes = len(class_names)

    # Print the class names and the total number of classes
    print("Class Names: \n", class_names)
    print("Number of Classes:", num_classes)


    """ 3. Examining Class Distribution in the Animal-10 Dataset """
    # Get the number of samples in each class
    class_sizes = []
    for name in class_names:
        class_size = len(os.listdir(data_path + "/" + name))
        class_sizes.append(class_size)

    # Print the class distribution
    print("Class Distribution:\n", class_sizes)


    """ 4. Visualizing Class Distribution in the Animal-10 Dataset using a Pie Chart and Bar Graph """
    # pie_chart()

    # bar_chart()


    """ 5. Sampling and Creating Sampled Data Directory """

    # Set the path to the directory where the sampled data will be saved
    sampled_data_path = './sampled-data'
    # create_directory()

    # Get a list of class names from the sampled data directory
    class_names = sorted(os.listdir(sampled_data_path))

    # Get the number of samples in each class
    class_sizes = []
    for name in class_names:
        # Get the number of samples in the class directory
        class_size = len(os.listdir(os.path.join(sampled_data_path, name)))
        class_sizes.append(class_size)

    # Print the class distribution
    print("Class Distribution:\n", class_sizes)

    # sampled_pie_chart()


    """ 6. Data Preparation and Augmentation """
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
        validation_split=0.2)

    # Load training data from the specified directory and apply the generator
    # target_size: resizes the images to a specified size
    # class_mode: specifies the type of label encoding, binary for 2 classes
    # batch_size: specifies the number of samples per batch
    # shuffle: shuffles the data after each epoch
    # subset: specifies the subset of data to load, in this case, the training set
    train_data = data_generator.flow_from_directory(
        sampled_data_path,
        target_size=(256, 256),
        class_mode='binary',
        batch_size=32,
        shuffle=True,
        subset='training')

    # Load validation data from the specified directory and apply the generator
    # subset: specifies the subset of data to load, in this case, the validation set
    valid_data = data_generator.flow_from_directory(
        sampled_data_path,
        target_size=(256, 256),
        class_mode='binary',
        batch_size=32,
        shuffle=True,
        subset='validation')


    """ 7. Data Visualization """
    # plot_dataset()


    """ 8. Model Training """
    # Specify the name of the model as "ResNet152V2".
    name = "ResNet152V2"

    # Load the pre-trained ResNet152V2 model, freeze its weights and exclude its final classification layer.
    base_model = ResNet152V2(include_top=False, input_shape=(256, 256, 3), weights='imagenet')
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
        epochs=3, callbacks=cbs
    )


    """ 9. Download Model for later use in Android app """
    keras.models.save_model(resnet152V2, 'resnet152V2.pbtxt')
    converter = tensorflow.lite.TFLiteConverter.from_keras_model(model=resnet152V2)
    model_tflite = converter.convert()
    open("animalClassificationModel.tflite", "wb").write(model_tflite)

    print('++ End of code')
