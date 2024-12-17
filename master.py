import matplotlib.pyplot as plt
import numpy as np

# Dummy data for plotting notes on lines
lines = [1, 2, 3, 4]  # Lines where the notes are
notes = ['C4', 'E4', 'G4', 'B4']

# Plot notes on lines
plt.figure(figsize=(10, 6))

# Create a simple plot with notes labeled on each line
for i, note in enumerate(notes):
    plt.text(0.5, lines[i], note, fontsize=12, ha='center', va='center')

plt.yticks(lines)
plt.xlabel("Music Staff")
plt.ylabel("Lines")
plt.title("Sheet Music Notes in English")
plt.grid(True)
plt.show()
