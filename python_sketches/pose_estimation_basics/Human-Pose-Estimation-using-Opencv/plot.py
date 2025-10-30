import matplotlib.pyplot as plt
import csv

# Read the CSV file
filename = "forearm.csv"  # Change this to your CSV filename

frames = []
elbow_angles = []

with open(filename, "r") as file:
    # Read the content and split by ';\n'
    content = file.read()
    rows = content.strip().split(";\n")

    # Skip the header row
    for row in rows[500:3357]:
        if row:  # Skip empty rows
            values = row.split(";")
            if len(values) >= 2:
                frames.append(float(values[0]))
                elbow_angles.append(float(values[1]))

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(frames, elbow_angles, marker="o", linestyle="-", linewidth=2, markersize=4)
plt.xlabel("Frame", fontsize=12)
plt.ylabel("Elbow Angle", fontsize=12)
plt.title("Elbow Angle vs Frame", fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
