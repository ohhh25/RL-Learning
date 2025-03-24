import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Function to map Q-value to a color (red to green gradient)
def q_to_color(q_value, q_min, q_range):
    normalized = (q_value - q_min) / q_range  # Normalize to [0, 1]
    # Red (1, 0, 0) to Green (0, 1, 0)
    red = 1 - normalized
    green = normalized
    blue = 0
    return (red, green, blue)

def plot_q_table(q_table, actions, fname=None):
    # Get dimensions of the grid
    rows, cols, _ = q_table.shape
    fig, ax = plt.subplots(figsize=(cols * 2, rows * 2))

    # Find min and max Q-values for color mapping
    q_min = np.min(q_table)
    q_max = np.max(q_table)
    q_range = q_max - q_min if q_max != q_min else 1  # Avoid division by zero

    # For each cell in the grid
    for i in range(rows):
        for j in range(cols):
            # Base coordinates of the cell (bottom-left corner)
            x = j
            y = rows - 1 - i  # Flip i to match typical grid orientation (top-left is (0,0))

            # Center of the cell
            center_x, center_y = x + 0.5, y + 0.5

            # Define the four triangles for each action
            # Each triangle is defined by 3 points: the center of the cell and two points on the edges
            triangles = {
                'UP': [(center_x, center_y), (x, y + 1), (x + 1, y + 1)],  # Top edge
                'DOWN': [(center_x, center_y), (x, y), (x + 1, y)],        # Bottom edge
                'LEFT': [(center_x, center_y), (x, y), (x, y + 1)],        # Left edge
                'RIGHT': [(center_x, center_y), (x + 1, y), (x + 1, y + 1)]  # Right edge
            }

            # Plot each triangle with the corresponding Q-value color
            for idx, action in enumerate(actions):
                q_value = q_table[i, j, idx]
                color = q_to_color(q_value, q_min, q_range)
                triangle = patches.Polygon(triangles[action], facecolor=color, edgecolor='black')
                ax.add_patch(triangle)

    # Set axis limits and labels
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.set_xticklabels(range(cols))
    ax.set_yticklabels(range(rows)[::-1])  # Reverse y-axis labels to match grid
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    ax.grid(True)

    # Add a colorbar to show the Q-value scale
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=plt.Normalize(vmin=q_min, vmax=q_max))
    plt.colorbar(sm, ax=ax, label='Q-Value')

    plt.title('Q-Table Visualization')

    if fname:
        plt.savefig(fname)
    else:
        plt.show()

