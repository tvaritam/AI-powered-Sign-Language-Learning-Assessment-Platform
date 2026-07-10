# AI-powered-Sign-Language-Learning-Assessment-Platform

# Sign Language Platform - AI Preprocessing Pipeline

This repository contains the backend engine and AI preprocessing pipeline for a real-time Sign Language Recognition Platform. The system processes raw hand gesture datasets, extracts spatial landmarks, and validates data quality through an automated API administration endpoint.

---

##  Features & Architecture

The project implements a modular, service-oriented architecture divided into clear processing stages as required by the pipeline design guidelines:

1. **Landmark Extraction (`extract_landmarks.py`)**: Traverses image dataset folders, skips invalid subdirectories, and extracts structural coordinate data.
2. **Dataset Builder (`dataset_builder.py`)**: Formats extraction records with tracking headers ($x_0, y_0, z_0 \dots$ up to index 20) and builds the master `landmarks.csv`.
3. **Data Validation (`validator.py`)**: Evaluates data integrity, tracks failure logs, and generates quality metrics inside `dataset_report.json`.
4. **Service Orchestration (`preprocessing_service.py`)**: Directs the end-to-end preprocessing pipeline uniformly so it can be safely reused for future dataset additions.
5. **FastAPI Endpoint Router (`POST /api/v1/preprocess`)**: Allows system administrators to trigger structural dataset feature extraction remotely via HTTP requests.

---

Local Installation & Environment Setup
 1. Clone and Navigate
git clone <your-repository-url>
cd SignLanguagePlatform 

2. Configure the Python Sandbox Environment

# Windows (Git Bash / Command Prompt)
python -m venv venv
source venv/Scripts/activate

3. Install Required Dependencies
Bash
python -m pip install --upgrade pip setuptools wheel
pip install fastapi uvicorn pandas opencv-python numpy
Execution Instructions
Start your local Uvicorn development server from the project workspace root:

Bash
PYTHONPATH=backend ./venv/Scripts/python -m uvicorn app.main:app --reload
The server will start up locally at http://127.0.0.1:8000.

Testing the Pipeline via Swagger UI
Open your web browser and go to your interactive documentation dashboard:
http://127.0.0.1:8000/docs

Locate the POST /api/v1/preprocess endpoint bar.

Click Try it out and then press the blue Execute button.

Expected Successful Response (200 OK):
JSON
{
  "success": true,
  "message": "Dataset preprocessing completed successfully.",
  "data": {
    "images_processed": 10,
    "successful": 10,
    "failed": 0,
    "csv_file": "landmarks.csv",
    "success_percentage": "100.0%"
  }
}
 Generated ML Pipeline Artifacts
Once executed, the pipeline populates data files into the backend/app/ai/ workspace folder:

landmarks.csv: Contains the flattened 63 numerical spatial feature rows and label strings used as training matrices for future machine learning model development.

dataset_report.json: Provides an automated quality assurance logging audit showing overall pipeline validation status.
