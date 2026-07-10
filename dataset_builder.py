import pandas as pd
from app.ai.extract_landmarks import extract_landmarks_from_dataset

def build_dataset_csv(dataset_dir: str, csv_output: str):
    """
    Task 2: Call landmark extraction and save rows into an organized CSV structure.
    """
    # 1. Call landmark extraction function
    data_records, log_summary = extract_landmarks_from_dataset(dataset_dir)
    
    # 2. Create tracking column structural headers (x0, y0, z0 ... x20, y20, z20, label)
    columns = []
    for i in range(21):
        columns.extend([f"x{i}", f"y{i}", f"z{i}"])
    columns.append("label")
    
    # 3. Store every processed sample into landmarks.csv
    df = pd.DataFrame(data_records, columns=columns)
    df.to_csv(csv_output, index=False)
    
    return log_summary