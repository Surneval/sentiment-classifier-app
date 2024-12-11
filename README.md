# Sentiment Classifier App
### Made by Nadezhda Trusova, HSE, MDS-2
### For the course Large Scale Machine Learning-2

This repository contains a sentiment analysis web application that classifies movie reviews as **positive** or **negative**. It includes:

- **Model Training Scripts** (using `Hugging Face Transformers`, `PyTorch`, and `PEFT` for LoRA)
- **MLflow Tracking** for experiment logging
- **FastAPI Service** for model inference
- **Streamlit UI** for user interaction

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Setup and Installation](#setup-and-installation)
  - [Local Setup](#local-setup)
  - [Using Docker and Docker Compose](#using-docker-and-docker-compose)
- [Model Training](#model-training)
- [Running the Service](#running-the-service)
- [Running the UI](#running-the-ui)
- [Making Predictions](#making-predictions)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Overview

This application fine-tunes a DistilBERT model on the IMDB movie reviews dataset for sentiment classification. It applies Low-Rank Adaptation (LoRA) for more efficient training and logs results to MLflow. The final model is served via a FastAPI backend, and a Streamlit frontend provides a simple UI for user input.

## Features

- **Data Preprocessing:** Clean and split IMDB data into training and validation sets.
- **Training:** Uses `Hugging Face Transformers` and `PEFT` to fine-tune DistilBERT with LoRA.
- **Experiment Tracking:** Uses MLflow to log metrics and parameters.
- **Model Serving:** A FastAPI endpoint for inference.
- **User Interface:** A Streamlit web app to easily test sentiment predictions.

## Project Structure

```
.
├── data/                      # Data files (IMDB dataset, train/val splits)
├── model/                     # Final trained model files and tokenizer
├── outputs/                   # Training checkpoints
├── scripts/                   # Scripts for training, preprocessing, utilities
│   ├── preprocess_data.py
│   ├── train.py
│   ├── mlflow_callback.py
│   └── utils.py
├── service/                   # FastAPI service for model inference
│   ├── Dockerfile
│   ├── app.py
│   ├── model_inference.py
│   └── requirements.txt
├── ui/                        # Streamlit UI
│   ├── Dockerfile
│   ├── app.py
│   └── requirements.txt
├── docker-compose.yml
├── requirements.txt           # Root-level requirements for local dev environment
└── README.md
```

## Prerequisites

- **Python 3.11+**
- **pip** and **virtualenv** (or `venv`)
- **Docker & Docker Compose** (if running via containers)
- Sufficient memory and storage for model training and inference.

## Setup and Installation

### Local Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Surneval/sentiment-classifier-app.git
   cd sentiment-classifier-app
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv my_ml
   source my_ml/bin/activate
   ```

3. **Install the requirements:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Preprocess the data (optional, if not already done):**
   ```bash
   python scripts/preprocess_data.py
   ```
   This will create `train.csv` and `val.csv` in the `data` directory.

### Using Docker and Docker Compose

1. **Build images without cache (if needed):**
   ```bash
   docker-compose build --no-cache
   ```
   
2. **Run services:**
   ```bash
   docker-compose up -d
   ```
3. **Check services:**
   ```bash
   docker-compose ps
   ```
   
This will start both the API service and UI.

## Model Training

To train (or retrain) the model from scratch:

1. **Activate your virtual environment:**
   ```bash
   source my_ml/bin/activate
   ```

2. **Run the training script:**
   ```bash
   python scripts/train.py
   ```
   
This will:
- Preprocess the data (if not done).
- Train the model and save it to the `model/` directory.
- Log training metrics to `mlruns/` for MLflow tracking.

## Running the Service without Docker

If running locally without Docker:

1. **Activate your virtual environment:**
   ```bash
   source my_ml/bin/activate
   ```

2. **Start the FastAPI service:**
   ```bash
   cd service
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```

3. Access the API docs at:
   ```
   http://localhost:8000/docs
   ```

## Running the UI

If not using Docker:

1. **Activate the environment:**
   ```bash
   source my_ml/bin/activate
   ```

2. **Run the Streamlit UI:**
   ```bash
   cd ui
   streamlit run app.py
   ```

3. Access the UI at:
   ```
   http://localhost:8501
   ```

## Making Predictions

- **Via the API**:  
  Send a POST request to `http://localhost:8000/predict` with JSON:
  ```json
  {
    "review": "This movie was absolutely fantastic!"
  }
  ```
  
  You’ll receive a response:
  ```json
  {
    "sentiment": "positive"
  }
  ```

- **Via the UI**:
  Simply open `http://localhost:8501`, enter a review, and click "Predict".

## Troubleshooting

- **Large File Issues in Git:**  
  If you encounter Git LFS or large file issues, ensure you have Git LFS installed and tracking large files.

- **Missing Dependencies:**
  Run:
  ```bash
  pip install -r requirements.txt
  ```
  
- **Docker Errors:**
  - Ensure Docker Desktop is running.
  - Check `docker info` to confirm the daemon is active.
  - Try `docker-compose down` followed by `docker-compose up --build`.

- **Model or Missing Files:**
  If the model files are missing, rerun `python scripts/train.py` or download the model weights as instructed by the project’s documentation.

## License

This project is released under the [MIT License](LICENSE). You are free to modify and distribute this software as per the license conditions.

## UI Screenshots

![2024-12-11_14-08-51](https://github.com/user-attachments/assets/9709e2b8-d799-49e8-9d11-8e22a25d5941)
![2024-12-11_14-08-19](https://github.com/user-attachments/assets/8b55cb75-1367-4f86-b052-229ffbd15fd3)

## Backend Screenshots

![2024-12-11_14-10-04](https://github.com/user-attachments/assets/e0aa94fd-f13e-4fc8-89f1-cde2b1ae97bf)
![2024-12-11_14-10-33](https://github.com/user-attachments/assets/3f6038e0-3a0b-4d90-a963-4709e1a4e356)
