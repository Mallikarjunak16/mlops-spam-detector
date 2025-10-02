# mlops-spam-detector
An end-to-end MLOps pipeline for spam detection using Google Cloud.
# MLOps End-to-End SMS Spam Detector

This project is a complete, end-to-end MLOps pipeline for training, deploying, and serving an SMS spam detection model using Google Cloud Platform. The system is fully automated, from weekly model retraining to a live prediction API.

## Features

- **Automated Retraining**: A serverless function retrains the model on a weekly schedule using Cloud Scheduler.
- **Live Prediction API**: A separate serverless function provides a public HTTP endpoint to get real-time spam predictions.
- **Serverless Architecture**: Built entirely on serverless components (Cloud Functions, Cloud Storage) for scalability and cost-efficiency.
- **Version Controlled**: All source code is version controlled with Git and hosted on GitHub.
- **Web Frontend**: A simple HTML/JavaScript frontend to demonstrate a real-world use case for the prediction API.

## Architecture

The project consists of two main pipelines that work together through Google Cloud Storage:

1.  **The Training Pipeline ("Factory")**:
    `Cloud Scheduler (triggers weekly)` -> `retrain-spam-model (Cloud Function)` -> `Writes new model.pkl to GCS`

2.  **The Prediction Pipeline ("Storefront")**:
    `Mobile/Web App` -> `predict-spam (Cloud Function API)` -> `Reads latest model.pkl from GCS` -> `Returns JSON prediction`

## Technologies Used

- **Cloud Platform**: Google Cloud Platform (GCP)
- **Machine Learning**: Python, Pandas, Scikit-learn, XGBoost, Jupyter
- **Backend**: Serverless (Google Cloud Functions)
- **Storage**: Google Cloud Storage
- **Automation**: Google Cloud Scheduler
- **Version Control**: Git & GitHub

## Project Structure
