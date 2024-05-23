# Sign Language Recognition Project

## Overview

This project focuses on recognizing sign language using computer vision techniques and the MediaPipe Holistic model. The system leverages OpenCV for image processing, MediaPipe for landmark detection, and various machine learning models for sign recognition.

## Features

- **Holistic Model Integration**: Utilizes the MediaPipe Holistic model to detect and draw landmarks for the face, pose, and hands.
- **Real-time Detection**: Processes video frames in real-time to detect and visualize sign language gestures.
- **Custom Drawing Utilities**: Implements custom functions to draw landmarks and connections with various styles.

## Getting Started

### Prerequisites

- Python 3.x
- OpenCV
- NumPy
- Matplotlib
- MediaPipe

### Installation

1. **Clone the Repository**:

    ```bash
    git clone https://github.com/yourusername/sign-language-recognition.git
    cd sign-language-recognition
    ```

2. **Install Dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

    Ensure that the `requirements.txt` file contains all necessary dependencies:

    ```text
    opencv-python
    numpy
    matplotlib
    mediapipe
    ```

### Running the Notebook

1. **Launch Jupyter Notebook**:

    ```bash
    jupyter notebook
    ```

2. **Open and Run the Notebook**:

    - Open the `sign_language_project.ipynb` file.
    - Run the cells sequentially to execute the code.

## Project Structure

- **1. Import and Install Dependencies**:
    ```python
    import cv2
    import numpy as np
    import os
    from matplotlib import pyplot as plt
    import time
    import mediapipe as mp
    ```

- **2. Key Points using MP Holistic**:
    ```python
    mp_holistic = mp.solutions.holistic    # Holistic model
    mp_drawing = mp.solutions.drawing_utils  # Drawing utilities
    ```

- **3. Detection and Drawing Utilities**:
    ```python
    def mediapipe_detection(image, model):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = model.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, results

    def draw_landmarks(image, results):
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    def draw_style_landmarks(image, results):
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                                 mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                 mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1))
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    ```


Thank you for checking out this project! If you have any questions or suggestions, feel free to contact us or open an issue on GitHub.
