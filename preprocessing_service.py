import os
from app.ai.dataset_builder import build_dataset_csv
from app.ai.validator import generate_validation_report

class PreprocessingService:
    def __init__(self):
        self.dataset_dir = os.path.abspath("datasets")
        self.csv_output = os.path.abspath("backend/app/ai/landmarks.csv")
        self.report_output = os.path.abspath("backend/app/ai/dataset_report.json")

    def execute_preprocessing(self) -> dict:
        os.makedirs(os.path.dirname(self.csv_output), exist_ok=True)
        
        # Run Task 2 (which calls Task 1 internally)
        log_summary = build_dataset_csv(self.dataset_dir, self.csv_output)
        
        # Run Task 3 (Generates validation json report)
        report_data = generate_validation_report(log_summary, self.report_output)
        
        # Return the clean flat dict for FastAPI endpoint processing
        return report_data