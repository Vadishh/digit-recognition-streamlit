# Handwritten Digit Recognition

A simple Streamlit web app that recognizes handwritten digits using multiple deep learning models.  
The app allows users to draw a digit directly on the screen and compares predictions from Perceptron, ANN, and CNN models.

## Features

- Interactive drawing canvas for digit input
- Real-time digit prediction
- Comparison between Perceptron, ANN, and CNN models
- Final prediction based on CNN output
- Clean and simple Streamlit interface

## Tech Stack

- Python
- Streamlit
- TensorFlow / Keras
- NumPy
- Pillow
- streamlit-drawable-canvas

## Models Used

The app uses three trained models:

- Perceptron model
- Artificial Neural Network model
- Convolutional Neural Network model

The CNN model is used for the final prediction because it performs better on image-based digit recognition tasks.

## Project Structure

```text
DIGIT_pred/
│
├── app.py
├── perceptron_model_digit.keras
├── ann_model_digit.keras
├── cnn_model_digit.keras
├── requirements.txt
└── README.md
