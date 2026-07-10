from fastapi import APIRouter, Depends
from app.services.preprocessing_service import PreprocessingService

router = APIRouter()

@router.post("/preprocess")
def trigger_preprocessing(service: PreprocessingService = Depends()):
    metrics = service.execute_preprocessing()
    
    return {
        "success": True,
        "message": "Dataset preprocessing completed successfully.",
        "data": {
            "images_processed": metrics.get("total_processed", 0),
            "successful": metrics.get("successful_detections", 0),
            "failed": metrics.get("no_hand_detected", 0) + metrics.get("corrupted_or_unreadable", 0),
            "csv_file": "landmarks.csv",
            "success_percentage": f"{metrics.get('success_percentage', 0.0)}%"
        }
    }