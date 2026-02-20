# Pegmatite_Finder_Dr.Mutlu-Zeybek
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Set up the figure and axis with a clean look
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.5

fig, ax = plt.subplots(1, 1, figsize=(16, 9))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.set_aspect('equal')

# --- Define Colors with better transparency and shading ---
colors = {
    'L1_sed': '#c2b280',  # Sandstone / Sediment
    'L2_meta': '#98b87c',  # Greenschist / Metamorphic
    'L3_granite': '#d56c5b', # Pink Granite
    'fault': '#2c3e50',     # Dark grey
    'pegmatite': '#8e44ad',  # Purple
    'background': '#ecf0f1'  # Light grey
}

# --- 1. Draw Background Lithological Units (Layers) ---
# L1 - Sedimentary (West)
ax.add_patch(plt.Rectangle((0, 0), 2, 6, color=colors['L1_sed'], alpha=0.8, ec='black', lw=1, label='_nolegend_'))
ax.text(1, 5.5, 'L1', ha='center', va='center', fontsize=16, fontweight='bold', color='black', alpha=0.7)

# L2 - Metamorphic (West of Granite)
ax.add_patch(plt.Rectangle((2, 0), 2, 6, color=colors['L2_meta'], alpha=0.8, ec='black', lw=1, label='_nolegend_'))
ax.text(3, 5.5, 'L2', ha='center', va='center', fontsize=16, fontweight='bold', color='black', alpha=0.7)

# L3 - Granite (Central)
ax.add_patch(plt.Rectangle((4, 0), 2, 6, color=colors['L3_granite'], alpha=0.8, ec='black', lw=2, label='_nolegend_'))
ax.text(5, 5.5, 'L3', ha='center', va='center', fontsize=16, fontweight='bold', color='white', alpha=0.9)

# L2 - Metamorphic (East of Granite)
ax.add_patch(plt.Rectangle((6, 0), 2, 6, color=colors['L2_meta'], alpha=0.8, ec='black', lw=1, label='_nolegend_'))
ax.text(7, 5.5, 'L2', ha='center', va='center', fontsize=16, fontweight='bold', color='black', alpha=0.7)

# L1 - Sedimentary (East)
ax.add_patch(plt.Rectangle((8, 0), 2, 6, color=colors['L1_sed'], alpha=0.8, ec='black', lw=1, label='_nolegend_'))
ax.text(9, 5.5, 'L1', ha='center', va='center', fontsize=16, fontweight='bold', color='black', alpha=0.7)

# --- 2. Draw Faults (F1 to F9) ---
fault_x_positions = np.arange(1, 10)
for i, x in enumerate(fault_x_positions):
    fault_id = i + 1
    ax.plot([x, x], [0.5, 5.5], color=colors['fault'], linestyle='-', linewidth=3, alpha=0.9, zorder=5)
    # Add fault label
    ax.text(x, 0.2, f'F{fault_id}', ha='center', va='center', fontsize=12, fontweight='bold', 
            color='white', bbox=dict(facecolor=colors['fault'], edgecolor='none', pad=2))

# --- 3. Draw Pegmatite Targets (P1 to P9) ---
pegmatite_y_positions = [1.5, 2.5, 3.5, 4.5]  # Different y-levels for visual clarity
target_details = [
    (1, 'P1'), (2, 'P2'), (3, 'P3'), (4, 'P4'), 
    (5, 'P5'), (6, 'P6'), (7, 'P7'), (8, 'P8'), (9, 'P9')
]

# Style for the pegmatite markers
marker_style = dict(s=400, color=colors['pegmatite'], edgecolor='black', 
                    linewidth=2, alpha=0.9, zorder=10)

# Scatter plot for all targets (at slightly varying heights to avoid overlap with fault lines)
y_positions = [3.0, 2.0, 4.0, 1.5, 3.5, 2.5, 4.5, 3.0, 2.0]  # Manually assigned for clarity
for i, (x, label) in enumerate(target_details):
    ax.scatter(x, y_positions[i], **marker_style)
    ax.text(x, y_positions[i], label, ha='center', va='center', fontsize=12, fontweight='bold', color='white', zorder=11)

# --- 4. Add a subtle grid and labels for coordinates ---
ax.set_xticks(np.arange(0, 11, 1))
ax.set_yticks(np.arange(0, 7, 1))
ax.set_xticklabels([f'x{i}' for i in range(0, 11)], fontsize=9)
ax.set_yticklabels([f'y{i}' for i in range(0, 7)], fontsize=9)
ax.grid(True, linestyle=':', alpha=0.3)
ax.set_xlabel('X Coordinate (Distance)', fontsize=14, fontweight='bold')
ax.set_ylabel('Y Coordinate (Stratigraphic Level)', fontsize=14, fontweight='bold')

# --- 5. Create a Legend ---
legend_elements = [
    mpatches.Patch(color=colors['L1_sed'], label='Sedimentary (L1)', alpha=0.8),
    mpatches.Patch(color=colors['L2_meta'], label='Metamorphic (L2)', alpha=0.8),
    mpatches.Patch(color=colors['L3_granite'], label='Granitic (L3)', alpha=0.8),
    plt.Line2D([0], [0], color=colors['fault'], lw=3, label='Fault (F1-F9)'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['pegmatite'], 
               markersize=12, markeredgecolor='black', label='Pegmatite Target (P1-P9)')
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=12, framealpha=0.95, edgecolor='black')

# --- 6. Title and Annotations ---
ax.set_title('The ZEYBEK-4 Model: Geometry-Driven Targeting of Pegmatites', 
             fontsize=18, fontweight='bold', pad=20)
ax.text(5, 5.8, 'Idealized Map View of Fault-Lithology Intersections', 
        ha='center', fontsize=12, style='italic', color='dimgrey')

# Add a scale bar (conceptual)
scale_bar_text = 'Conceptual Coordinate System'
ax.text(9.5, 0.1, scale_bar_text, ha='right', va='bottom', fontsize=9, color='grey')

plt.tight_layout()
plt.show()

#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
