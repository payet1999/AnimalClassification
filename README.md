# AnimalClassification

<p align="center">
  <img src="dog.png" alt="AnimalClassification Logo" width="120" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Kotlin-1.8%2B-7F52FF?style=for-the-badge&logo=kotlin&logoColor=white" alt="Kotlin" />
  <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/TensorFlow%20Lite-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow Lite" />
  <img src="https://img.shields.io/badge/Android-CameraX-3DDC84?style=for-the-badge&logo=android&logoColor=white" alt="Android CameraX" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License" />
</p>

A single activity Android app using a custom TensorFlow Lite model to classify animals from camera pictures. If a dog is
detected, the app also predicts the dog breed.

## Repository Background

This project was created as a POC of an Android app with machine learning algorithms integrated directly on-device.
The code was created in 2023 with only a few months of professional experience as a sandbox for personal experimentation.
It is now public to serve as part of a general project portfolio.

## Training Datasets & ML Model Metrics

### Datasets

- **Animals Dataset**: https://www.kaggle.com/datasets/alessiocorrado99/animals10?resource=download
- **Dogs Dataset**: https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset + Shiba Inu (30 pictures)

### ML Model Metrics

- **Animals Model**: loss: 0.2883 - accuracy: 0.9055 - val_loss: 0.3348 - val_accuracy: 0.8844
- **Dogs Model**: loss: 1.3236 - accuracy: 0.6195 - val_loss: 1.4665 - val_accuracy: 0.5954

## How to Run

The repository includes only the essential files required to replicate the project. Larger dataset directories and compiled model files were omitted to reduce repository size. 

To build and run the application locally:
1. Download the original datasets from the links above.
2. Install the required Python prerequisites (`pip install -r python/requirements.txt`).
3. Run both Python training scripts to generate the `.tflite` model files.
4. Create or open the project in Android Studio.
5. Install NDK / CMake dependencies if required by TensorFlow Lite in Android Studio.
6. Copy the generated `.tflite` files into `android/app/src/main/assets/`.
7. Move `dog_breeds_labels.txt` to `android/app/src/main/res/raw/`.
8. Build the APK and deploy it to your test device.

## Expected Folder Structure

```text
├── android/
│   ├── gradlew
│   ├── gradle/
│   │   └── wrapper/
│   └── app/
│       ├── build.gradle
│       └── src/
│           └── main/
│               ├── AndroidManifest.xml
│               ├── assets/       # AnimalClassificationModelOptimized.tflite, DogBreedClassificationModel.tflite
│               ├── java/         # Kotlin source code (MainActivity.kt)
│               └── res/          # Layouts (activity_main.xml) & raw resources
├── python/
│   ├── requirements.txt
│   ├── animals_model_training_script.py
│   └── dogs_model_training_script.py
├── datasets/
│   ├── animals_dataset/
│   │   ├── dog/
│   │   ├── horse/
│   │   └── ...
│   └── dogs_dataset/
│       ├── Afghan_hound/
│       ├── toy_terrier/
│       └── ...
├── README.md
└── .gitignore