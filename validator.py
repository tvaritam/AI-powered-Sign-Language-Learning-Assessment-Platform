import json

def generate_validation_report(log_summary: dict, report_output: str):
    """
    Task 3: Introduce quality checks and save the structural dataset metrics.
    """
    total = log_summary["total_processed"]
    success = log_summary["successful_detections"]
    
    success_percentage = 0.0
    if total > 0:
        success_percentage = round((success / total) * 100, 1)
        
    report_data = {
        "total_processed": total,
        "successful_detections": success,
        "no_hand_detected": log_summary["no_hand_detected"],
        "corrupted_or_unreadable": log_summary["corrupted_or_unreadable"],
        "success_percentage": success_percentage
    }
    
    with open(report_output, "w") as f:
        json.dump(report_data, f, indent=4)
        
    return report_data