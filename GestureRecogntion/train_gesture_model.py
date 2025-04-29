import os
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
from fpdf import FPDF
import pandas as pd

# Directory setup
src_root = 'g_data'
dest_root = 'gesture_data_split'
val_size = 0.15
test_size = 0.15

# Prepare summary dictionary
summary = defaultdict(lambda: {'train': 0, 'val': 0, 'test': 0})
classes = ['blink', 'smile']

# Function to copy and count
for cls in classes:
    class_path = os.path.join(src_root, cls)
    if not os.path.isdir(class_path):
        print(f"Folder not found: {class_path}")
        continue

    images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not images:
        print(f"No images in {class_path}")
        continue

    train_val, test = train_test_split(images, test_size=test_size, random_state=42)
    train, val = train_test_split(train_val, test_size=val_size / (1 - test_size), random_state=42)

    summary[cls]['train'] = len(train)
    summary[cls]['val'] = len(val)
    summary[cls]['test'] = len(test)

    for split_name, split_files in zip(['train', 'val', 'test'], [train, val, test]):
        split_dir = os.path.join(dest_root, split_name, cls)
        os.makedirs(split_dir, exist_ok=True)
        for file in tqdm(split_files, desc=f"Copying {split_name}/{cls}", unit="file"):
            src_path = os.path.join(class_path, file)
            dst_path = os.path.join(split_dir, file)
            shutil.copy(src_path, dst_path)

# Bar plot of image distribution
categories = list(summary.keys())
train_counts = [summary[label]['train'] for label in categories]
val_counts = [summary[label]['val'] for label in categories]
test_counts = [summary[label]['test'] for label in categories]

x = range(len(categories))
plt.figure(figsize=(10, 6))
plt.bar(x, train_counts, width=0.25, label='Train', align='center')
plt.bar([i + 0.25 for i in x], val_counts, width=0.25, label='Val', align='center')
plt.bar([i + 0.5 for i in x], test_counts, width=0.25, label='Test', align='center')
plt.xticks([i + 0.25 for i in x], categories)
plt.ylabel("Number of Images")
plt.title("Dataset Split Summary")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
chart_path = "gesture_data_split_summary.png"
plt.savefig(chart_path)

# Create summary PDF
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)
pdf.cell(200, 10, txt="Gesture Dataset Split Summary", ln=True, align='C')
pdf.ln(10)

for cls in classes:
    pdf.cell(200, 10, txt=f"{cls.capitalize()} - Train: {summary[cls]['train']}, Val: {summary[cls]['val']}, Test: {summary[cls]['test']}", ln=True)

pdf.image(chart_path, x=10, y=None, w=190)
pdf_path = "summary.pdf"
pdf.output(pdf_path)

# Display final counts as a DataFrame
summary_df = pd.DataFrame(summary).T
print("\n Summary Table:\n")
summary_df

pdf_path, chart_path
