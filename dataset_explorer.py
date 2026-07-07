import os
import csv

def analyze_asl_dataset(dataset_path):
    """Parses local folder image distribution structures to export statistical data."""
    if not os.path.exists(dataset_path):
        print(f"[ERROR] The path folder '{dataset_path}' does not exist.")
        return

    print(f"[RUNNING] Analyzing dataset folders inside: {dataset_path} ...")
    
    class_counts = {}
    total_images = 0

    # Read each gesture folder directory inside dataset
    for item in os.listdir(dataset_path):
        item_path = os.path.join(dataset_path, item)
        if os.path.isdir(item_path):
            # Count common image file extension assets
            images = [f for f in os.listdir(item_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            class_counts[item] = len(images)
            total_images += len(images)

    if not class_counts:
        print("[WARNING] No image classification folders discovered.")
        return

    # Calculate extremes
    largest_class = max(class_counts, key=class_counts.get)
    smallest_class = min(class_counts, key=class_counts.get)
    total_classes = len(class_counts)

    # Export to summary CSV
    csv_filename = "asl_dataset_summary.csv"
    with open(csv_filename, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        
        # Section 1: Meta Summary
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Total Gesture Classes", total_classes])
        writer.writerow(["Total Dataset Images", total_images])
        writer.writerow(["Largest Class Designation", f"{largest_class} ({class_counts[largest_class]} images)"])
        writer.writerow(["Smallest Class Designation", f"{smallest_class} ({class_counts[smallest_class]} images)"])
        writer.writerow([]) # Blank structural divider row
        
        # Section 2: Distribution Breakdown Breakdown List
        writer.writerow(["Gesture Class Label", "Image Volume Count"])
        for gesture, count in sorted(class_counts.items()):
            writer.writerow([gesture, count])

    print(f"[SUCCESS] Exploration metrics exported completely to: {csv_filename}")

if __name__ == "__main__":
    # Point this to whatever local folder holds your dataset alphabet directories 
    # Example relative path: "backend/app/ai/dataset" or similar path
    TARGET_DATASET_FOLDER = "datasets" 
    analyze_asl_dataset(TARGET_DATASET_FOLDER)