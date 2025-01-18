
# Sign Language Recognition using [CNN,openCv,TensorFlow,Streamlit]

This repository contains the code for a Sign Language Recognition project. The project aims to recognize and interpret sign language gestures using deep learning techniques.

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Dataset](#dataset)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Streamlit Application](#streamlit-application)
7. [Model Architecture](#model-architecture)
8. [Training](#training)
9. [Evaluation](#evaluation)
10. [Results](#results)
11. [Contributing](#contributing)
12. [License](#license)
13. [References](#references)

## Overview

This project focuses on developing a model to recognize sign language gestures from images by using the concepts of convolutional neural networks. The model is trained to identify various hand gestures that correspond to different signs in the sign language.

## Features

- **Real-time Sign Language Recognition**: The model can recognize gestures in real-time using a webcam .
- **High Accuracy**: Achieved through the use of advanced deep learning techniques.
- **User-friendly Interface**: An intuitive interface for users to interact with the model.

## Dataset

The dataset used in this project was custom-prepared by capturing images of each gesture and assigning corresponding labels. The process involved manually collecting data for various sign language gestures, ensuring comprehensive coverage of the targeted vocabulary.

The dataset structure includes:

Training Data: Organized in directories based on gesture classes.

Labels: Stored in sign_language_data, which maps class indices to labels.

Test Data: Placed in the data/test directory for evaluation purposes.

This custom dataset can be replaced or expanded to include additional gestures or languages.

## Installation

Follow these steps to set up and run the project on your local machine:

### Prerequisites

Make sure you have the following installed:

- **Python 3.10**
- **pip** (Python package installer)
- **TensorFlow** or **PyTorch** (depending on the framework used)
- **OpenCV** (for video processing)
- **Flask** or **Streamlit** (for the web interface)

### Steps

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/gowtham611/sign_language.git
   cd sign_language
   ```

2. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

To run the application:

```bash
python example.py
```

##Streamlit Application

The Streamlit app provides the following features:

**Homepage**: Introduction to the project and instructions for users.

**Real-Time Recognition**: Recognizes gestures in real-time using a webcam.

**Interactive Games**: Fun games that involve sign language gestures.

**Chatbot**: An integrated chatbot for user assistance and interaction.

![image](https://github.com/user-attachments/assets/7eb79764-15e1-4e3b-a9fd-d7e0580e1a38)

![image](https://github.com/user-attachments/assets/98115029-1983-479e-bacd-1ab04b738e4d)

![image](https://github.com/user-attachments/assets/774b38e3-251a-4faf-ba22-2fae873ea1ec)

![image](https://github.com/user-attachments/assets/95f825a0-5b5c-4b83-92db-32242cba7798)

![image](https://github.com/user-attachments/assets/e254d85d-ce5e-4b94-a356-19bc5481ab91)


## Model Architecture

Describe the architecture of your model here. For example:

- **Convolutional Neural Network (CNN)**: Used for image-based gesture recognition.
- **Transfer Learning**: Using pre-trained models like VGG16, ResNet, etc., for better performance.(you can use pretrained model if you want to avoid training from scratch but i trained the model from scratch)
- **Pooling Layers**:To reduce spatial dimensions.
- **Fully Connected Layers**: For classification.
- **Convolutional Layers**: For feature extraction
The model is implemented in the datacollection.ipynb file

## Training

Outline the training process here. Include information on data augmentation, optimization techniques, loss functions, etc.

## Evaluation

Explain how the model is evaluated. Include metrics such as accuracy, precision, recall, F1-score, etc.

## Results

Provide details on the results achieved by the model. Include examples of predictions, accuracy scores.
The results section highlights the performance of the model:
Accuracy: 93%

Sample Predictions: Visual examples of input gestures and their predicted labels.
It basically contains a real time recognition that is used to identify the gestures.

![image](https://github.com/user-attachments/assets/bbb62ed3-dcca-4534-808a-e4443691aa7f)


## Contributing

If you would like to contribute to this project, please follow these guidelines:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -am 'Add new feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Create a new Pull Request.

## License

This project currently does not have a license. If you intend to make your work publicly available and reusable, consider adding an open-source license. For example:

MIT License (permissive and widely used)

Apache License 2.0 (also permissive with patent rights)

GPLv3 (requires derivative works to be open-sourced)

Refer to Choose a License for guidance on selecting the appropriate license for your project.

## References

List any references or resources used in the project. For example:

1. TensorFlow Documentation: https://www.tensorflow.org/
2. OpenCV Documentation: https://opencv.org/
3. Streamlit Documentation: https://www.streamlit.org/

