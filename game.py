import math
import pandas as pd
import matplotlib.pyplot as plt

class MockData():

    class MockClass():
        def __init__(self, name, major, code, x, y):
            self.name = name
            self.major = major
            self.code = str(code)
            self.x = x - 0.5
            self.y = y - 0.5
            self.angle = math.atan2(self.y, self.x)


    def __init__(self):
        classes = self.get_classes()
    
    def get_classes(self):
        df = pd.read_csv("mock_data/classes.csv")
        self.all_classes = []

        for index, row in df.iterrows():
            self.all_classes.append(self.MockClass(row['name'], row['major'], row['code'], row['x'], row['y']))
        
        return self.all_classes

    def display(self):
        mock_classes = self.all_classes

        # 3. Setup the Matplotlib figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        plt.subplots_adjust(hspace=0.4)

        # --- PLOT 1: VECTOR/EMBEDDING DISPLAY (X, Y) ---
        ax1.set_title("Class Skill Embedding Vectors (Writing vs. Logic Intensity)", fontsize=14)
        ax1.set_xlabel("X-Axis: Writing Intensity (e.g., Argumentation, Essay Structure)")
        ax1.set_ylabel("Y-Axis: Logic Intensity (e.g., Mathematical Rigor, Problem Solving)")
        ax1.axhline(0, color='gray', linestyle='--', linewidth=0.5)
        ax1.axvline(0, color='gray', linestyle='--', linewidth=0.5)
        ax1.set_xlim(-1, 1.0)
        ax1.set_ylim(-1, 1.0)
        ax1.set_aspect('equal', adjustable='box')

        # Draw the vectors (using quiver for clear direction)
        ax1.quiver([0] * len(mock_classes), [0] * len(mock_classes), 
                [c.x for c in mock_classes], [c.y for c in mock_classes], 
                angles='xy', scale_units='xy', scale=1, alpha=0.6, width=0.005)

        # Add the end points and labels
        for c in mock_classes:
            # Scatter plot for the end point
            ax1.scatter(c.x, c.y, s=100, zorder=5)
            # Label the end point
            ax1.annotate(c.name + c.code, (c.x, c.y), 
                        textcoords="offset points", 
                        xytext=(5, 5), 
                        ha='center', 
                        fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))


        # --- PLOT 2: ANGLE DISTRIBUTION (Line Graph) ---
        angles = [c.angle for c in mock_classes]
        labels = [c.name + c.code for c in mock_classes]

        ax2.set_title("Class Distribution by Vector Angle (-180° to 180°)", fontsize=14)
        ax2.set_xlabel("Angle (Degrees)")
        ax2.set_yticks([]) # Hide the y-axis (it's a 1D distribution)
        ax2.set_xlim(-1, 1)
        ax2.axvline(0, color='red', linestyle='-', linewidth=1, alpha=0.5) # Mark 0 degrees (Positive X-axis)
        ax2.axvline(1, color='red', linestyle='--', linewidth=1, alpha=0.5) # Mark 180 degrees (Negative X-axis)
        ax2.axvline(0, color='red', linestyle='--', linewidth=1, alpha=0.5) # Mark -180 degrees (Negative X-axis)

        # Plot the points slightly above the line for visibility
        y_pos = [0] * len(angles)
        ax2.scatter(angles, y_pos, s=100, zorder=5) 
        ax2.plot([-1, 1], [0, 0], 'k-', linewidth=1.5) # The main angle line

        # Add the labels (using a simple alternating offset to avoid overlap)
        label_offset = [5, -15] # Alternating offsets
        for i, c in enumerate(mock_classes):
            ax2.annotate(c.major + c.code, (c.angle, 0), 
                        textcoords="offset points", 
                        xytext=(0, label_offset[i % 2]), 
                        ha='center', 
                        fontsize=9, 
                        arrowprops=dict(arrowstyle="-", connectionstyle="arc3,rad=0", color="black", alpha=0.5, linewidth=0.5))

        plt.show()



data = MockData()
data.get_classes()
data.display()

