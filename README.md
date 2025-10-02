# MLOps End-to-End SMS Spam Detector

This project is a complete, end-to-end MLOps pipeline for training, deploying, and serving an SMS spam detection model using Google Cloud Platform. The system is fully automated, from weekly model retraining to a live prediction API.

## 1. Problem Statement

SMS spam is a significant problem, ranging from annoying advertisements to malicious phishing and financial fraud attempts. Manually filtering these messages is impractical. There is a need for an intelligent, automated system that can accurately identify and flag spam messages in real-time to protect users.

## 2. The Solution

This project implements a complete MLOps pipeline that provides a scalable and auto-improving solution to this problem. The core of the solution is a machine learning model that is automatically retrained on a weekly basis to adapt to new spam trends. This model is served via a live, public API that can be integrated into any application (e.g., messaging apps, banking apps) to provide instant spam-checking functionality.

## 3. Key Features

- **Automated Retraining**: A serverless function retrains the model on a weekly schedule using Cloud Scheduler.
- **Live Prediction API**: A separate serverless function provides a public HTTP endpoint to get real-time spam predictions.
- **Serverless Architecture**: Built entirely on serverless components (Cloud Functions, Cloud Storage) for scalability and cost-efficiency.
- **Version Controlled**: All source code is version controlled with Git and hosted on GitHub.
- **Web Frontend**: A simple HTML/JavaScript frontend to demonstrate a real-world use case for the prediction API.

## 4. Model Performance

The final XGBoost model was evaluated on a test set. The performance metrics below demonstrate its effectiveness in correctly identifying spam while maintaining a low false-positive rate.

- **Accuracy**: XX.XX%
- **Precision (Spam)**: X.XX
- **Recall (Spam)**: X.XX
- **F1-Score (Spam)**: X.XX

## 5. Architecture

The project consists of two main pipelines that work together through Google Cloud Storage:

1.  **The Training Pipeline ("Factory")**:
    `Cloud Scheduler (triggers weekly)` -> `retrain-spam-model (Cloud Function)` -> `Writes new model.pkl to GCS`

2.  **The Prediction Pipeline ("Storefront")**:
    `Mobile/Web App` -> `predict-spam (Cloud Function API)` -> `Reads latest model.pkl from GCS` -> `Returns JSON prediction`

## 6. Technologies Used

- **Cloud Platform**: Google Cloud Platform (GCP)
- **Machine Learning**: Python, Pandas, Scikit-learn, XGBoost, Jupyter
- **Backend**: Serverless (Google Cloud Functions)
- **Storage**: Google Cloud Storage
- **Automation**: Google Cloud Scheduler
- **Version Control**: Git & GitHub

## 7. Project Structure
