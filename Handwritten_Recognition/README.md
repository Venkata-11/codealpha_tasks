# Handwritten Character Recognition

A machine learning web application developed as part of my CodeAlpha internship.

## Project Overview

This application recognizes handwritten digits, individual English characters, and complete handwritten text.

## Features

### 1. Digit Recognition
- Recognizes handwritten digits from 0 to 9.
- Supports drawing digits and uploading images.
- Uses a Convolutional Neural Network (CNN).

### 2. Character Recognition
- Recognizes individual handwritten English letters from A to Z.
- Uses a CNN trained on the EMNIST Letters dataset.

### 3. Handwritten Line Recognition
- Recognizes complete handwritten words and sentences from uploaded images.
- Uses a Convolutional Recurrent Neural Network (CRNN) with Connectionist Temporal Classification (CTC) decoding.
- Trained using the IAM Handwriting dataset.

## Technologies Used

- Python
- TensorFlow
- Streamlit
- NumPy
- OpenCV
- Pillow

## Project Structure

- `app.py` – Main Streamlit application.
- `models/` – Trained machine learning models.
- `data/` – Required datasets and vocabulary files.
- `src/` – Source code and supporting scripts.
- `notebooks/` – Jupyter notebooks.
- `results/` – Training and evaluation results.
- `requirements.txt` – Required Python libraries.

## How to Run

1. Install the required libraries:

   `pip install -r requirements.txt`

2. Run the application:

   `streamlit run app.py`

## Note

This project was developed for educational purposes as part of the CodeAlpha Machine Learning Internship.