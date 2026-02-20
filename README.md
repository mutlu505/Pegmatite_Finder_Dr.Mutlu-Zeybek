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
"""
ZEYBEK-4 Model: A Rule-Based Expert System for the Systematic, 
Geometry-Driven Targeting of Fault-Controlled Pegmatites

Author: Mutlu ZEYBEK
Based on the paper: "The ZEYBEK-4 Model: A Rule-Based Expert System for the 
Systematic, Geometry-Driven Targeting of Fault-Controlled Pegmatites"

License: Open source for academic and commercial use with attribution
Repository: https://github.com/mutlu505/Pegmatite_Finder_Dr.Mutlu-Zeybek
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon, box
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PART 1: CORE MODEL CLASSES AND FUZZY LOGIC ENGINE
# ============================================================================

class FuzzyMembership:
    """
    Fuzzy membership functions for handling geological uncertainty.
    Implements trapezoidal membership functions for various input parameters.
    """
    
    @staticmethod
    def fault_proximity(distance_m):
        """
        Membership function for fault proximity.
        
        Args:
            distance_m: Distance to nearest fault in meters
            
        Returns:
            float: Membership value [0, 1]
        """
        if distance_m <= 100:
            return 1.0
        elif 100 < distance_m < 500:
            return 1 - ((distance_m - 100) / 400)
        else:
            return 0.0
    
    @staticmethod
    def contact_proximity(distance_m):
        """
        Membership function for lithological contact proximity.
        
        Args:
            distance_m: Distance to nearest lithological contact in meters
            
        Returns:
            float: Membership value [0, 1]
        """
        if distance_m <= 50:
            return 1.0
        elif 50 < distance_m < 250:
            return 1 - ((distance_m - 50) / 200)
        else:
            return 0.0
    
    @staticmethod
    def fault_kinematic_favorability(fault_type, kinematic_regime='dilational'):
        """
        Membership function for fault kinematic favorability.
        
        Args:
            fault_type: 'strike-slip', 'normal', or 'reverse'
            kinematic_regime: 'dilational', 'transtensional', 'neutral', 
                             'transpressional', or 'contractional'
        
        Returns:
            float: Membership value [0, 1]
        """
        # Base values by fault type
        type_values = {
            'strike-slip': 0.9,
            'normal': 0.8,
            'reverse': 0.5,
            'unknown': 0.4
        }
        
        # Kinematic modifiers
        kinematic_modifiers = {
            'dilational': 1.1,
            'transtensional': 1.0,
            'neutral': 0.8,
            'transpressional': 0.5,
            'contractional': 0.3
        }
        
        base = type_values.get(fault_type.lower(), 0.4)
        modifier = kinematic_modifiers.get(kinematic_regime.lower(), 0.8)
        
        # Cap at 1.0
        return min(base * modifier, 1.0)
    
    @staticmethod
    def host_rock_fertility(lithology, metamorphic_grade=None):
        """
        Membership function for host rock fertility.
        
        Args:
            lithology: Rock type classification
            metamorphic_grade: 'amphibolite', 'greenschist', etc.
        
        Returns:
            float: Membership value [0, 1]
        """
        fertility_scores = {
            'amphibolite_facies_metapelite': 1.0,
            'greenschist_facies_metapelite': 0.9,
            'amphibolite': 0.8,
            'mafic_metavolcanic': 0.6,
            'granite': 0.4,
            'sedimentary': 0.2,
            'unknown': 0.1
        }
        
        # Construct key from inputs
        if metamorphic_grade and 'metapelite' in lithology.lower():
            key = f"{metamorphic_grade.lower()}_metapelite"
        elif 'amphibolite' in lithology.lower():
            key = 'amphibolite'
        elif 'metavolcanic' in lithology.lower() or 'mafic' in lithology.lower():
            key = 'mafic_metavolcanic'
        elif 'granite' in lithology.lower() or 'granitic' in lithology.lower():
            key = 'granite'
        elif 'sedimentary' in lithology.lower():
            key = 'sedimentary'
        else:
            key = 'unknown'
        
        return fertility_scores.get(key, 0.1)


class Zeybek4Model:
    """
    Main ZEYBEK-4 Model class implementing the rule-based expert system
    for pegmatite targeting.
    """
    
    # Target types with descriptions and base probabilities
    TARGET_TYPES = {
        'P1': {
            'name': 'Distal Sedimentary (West)',
            'description': 'Fault intersection with sedimentary rocks in western domain',
            'base_probability': 0.78,
            'std_dev': 0.12,
            'primary_ref': 'Selway et al. (2005)'
        },
        'P2': {
            'name': 'Sedimentary-Metamorphic Contact (West)',
            'description': 'Triple junction of sedimentary rocks, metamorphic rocks, and fault',
            'base_probability': 0.64,
            'std_dev': 0.15,
            'primary_ref': 'Errandonea-Martin et al. (2022)'
        },
        'P3': {
            'name': 'Metamorphic (West)',
            'description': 'Fault intersection with metamorphic rocks west of granite',
            'base_probability': 0.82,
            'std_dev': 0.09,
            'primary_ref': 'Černý (1989a)'
        },
        'P4': {
            'name': 'Metamorphic-Granite Contact (West)',
            'description': 'Western contact between metamorphic rocks and granite, intersected by fault',
            'base_probability': 0.71,
            'std_dev': 0.11,
            'primary_ref': 'Xiong et al. (2025)'
        },
        'P5': {
            'name': 'Within Granite',
            'description': 'Fault intersection within granitic body',
            'base_probability': 0.42,
            'std_dev': 0.18,
            'primary_ref': 'Černý (1989a)'
        },
        'P6': {
            'name': 'Granite-Metamorphic Contact (East)',
            'description': 'Eastern contact between granite and metamorphic rocks, intersected by fault',
            'base_probability': 0.69,
            'std_dev': 0.13,
            'primary_ref': 'Xiong et al. (2025)'
        },
        'P7': {
            'name': 'Metamorphic (East)',
            'description': 'Fault intersection with metamorphic rocks east of granite',
            'base_probability': 0.79,
            'std_dev': 0.10,
            'primary_ref': 'Černý (1989a)'
        },
        'P8': {
            'name': 'Metamorphic-Sedimentary Contact (East)',
            'description': 'Contact between metamorphic and sedimentary rocks east of granite, intersected by fault',
            'base_probability': 0.61,
            'std_dev': 0.14,
            'primary_ref': 'Errandonea-Martin et al. (2022)'
        },
        'P9': {
            'name': 'Distal Sedimentary (East)',
            'description': 'Fault intersection with sedimentary rocks in eastern domain',
            'base_probability': 0.74,
            'std_dev': 0.16,
            'primary_ref': 'Selway et al. (2005)'
        }
    }
    
    # Rule weights from meta-analysis
    RULE_WEIGHTS = {
        'fault_proximity_100m': 2.0,
        'fault_type_strike_slip_dilational': 1.5,
        'host_rock_amphibolite_metapelite': 1.8,
        'contact_proximity_50m': 1.6,
        'multiple_fault_intersections': 1.4,
        'granite_proximity_2km': 1.3
    }
    
    def __init__(self, anatectic_flag=False):
        """
        Initialize the ZEYBEK-4 Model.
        
        Args:
            anatectic_flag: If True, down-weights granite proximity and 
                           up-weights host rock fertility for anatectic scenarios
        """
        self.anatectic_flag = anatectic_flag
        self.fuzzy = FuzzyMembership()
        self.targets = []
        self.lithology_data = None
        self.fault_data = None
        
    def load_lithology_data(self, lithology_gdf):
        """
        Load lithological data as GeoDataFrame.
        
        Args:
            lithology_gdf: GeoDataFrame with lithology polygons and attributes
        """
        required_cols = ['lithology', 'metamorphic_grade']
        for col in required_cols:
            if col not in lithology_gdf.columns:
                lithology_gdf[col] = 'unknown'
        
        self.lithology_data = lithology_gdf
        print(f"Loaded {len(lithology_gdf)} lithology polygons")
        
    def load_fault_data(self, fault_gdf):
        """
        Load fault data as GeoDataFrame.
        
        Args:
            fault_gdf: GeoDataFrame with fault lines and attributes
        """
        required_cols = ['fault_id', 'fault_type', 'kinematics', 'dip', 'dip_dir']
        for col in required_cols:
            if col not in fault_gdf.columns:
                if col == 'fault_id':
                    fault_gdf[col] = [f'F{i}' for i in range(1, len(fault_gdf)+1)]
                else:
                    fault_gdf[col] = 'unknown'
        
        self.fault_data = fault_gdf
        print(f"Loaded {len(fault_gdf)} fault traces")
    
    def calculate_fault_buffers(self, search_radius=500):
        """
        Create buffers around fault traces.
        
        Args:
            search_radius: Buffer distance in meters
            
        Returns:
            GeoDataFrame of fault buffers
        """
        if self.fault_data is None:
            raise ValueError("Fault data not loaded")
        
        buffers = []
        for idx, fault in self.fault_data.iterrows():
            buffer = fault.geometry.buffer(search_radius, cap_style=2)  # flat caps
            buffers.append({
                'geometry': buffer,
                'fault_id': fault['fault_id'],
                'fault_type': fault['fault_type'],
                'kinematics': fault['kinematics'],
                'dip': fault['dip'],
                'dip_dir': fault['dip_dir']
            })
        
        return gpd.GeoDataFrame(buffers, crs=self.fault_data.crs)
    
    def calculate_contact_buffers(self, search_radius=250):
        """
        Create buffers around lithological contacts.
        
        Args:
            search_radius: Buffer distance in meters
            
        Returns:
            GeoDataFrame of contact buffers
        """
        if self.lithology_data is None:
            raise ValueError("Lithology data not loaded")
        
        # Convert polygons to lines (contacts)
        boundaries = self.lithology_data.boundary
        contacts = gpd.GeoDataFrame(geometry=boundaries[~boundaries.is_empty], 
                                     crs=self.lithology_data.crs)
        
        # Buffer the contacts
        contact_buffers = contacts.buffer(search_radius)
        
        # Create GeoDataFrame
        buffer_gdf = gpd.GeoDataFrame(geometry=contact_buffers, crs=self.lithology_data.crs)
        buffer_gdf['buffer_type'] = 'contact'
        
        return buffer_gdf
    
    def identify_intersection_zones(self, fault_buffers, contact_buffers):
        """
        Identify zones where fault buffers and contact buffers intersect.
        
        Args:
            fault_buffers: GeoDataFrame of fault buffers
            contact_buffers: GeoDataFrame of contact buffers
            
        Returns:
            GeoDataFrame of intersection polygons
        """
        intersections = []
        
        for f_idx, fault in fault_buffers.iterrows():
            for c_idx, contact in contact_buffers.iterrows():
                if fault.geometry.intersects(contact.geometry):
                    intersection = fault.geometry.intersection(contact.geometry)
                    if not intersection.is_empty:
                        # Handle MultiPolygon
                        if intersection.geom_type == 'MultiPolygon':
                            for poly in intersection.geoms:
                                intersections.append({
                                    'geometry': poly,
                                    'fault_id': fault['fault_id'],
                                    'fault_type': fault['fault_type'],
                                    'kinematics': fault['kinematics'],
                                    'dip': fault['dip'],
                                    'dip_dir': fault['dip_dir']
                                })
                        else:
                            intersections.append({
                                'geometry': intersection,
                                'fault_id': fault['fault_id'],
                                'fault_type': fault['fault_type'],
                                'kinematics': fault['kinematics'],
                                'dip': fault['dip'],
                                'dip_dir': fault['dip_dir']
                            })
        
        if not intersections:
            return gpd.GeoDataFrame(columns=['geometry', 'fault_id', 'fault_type', 
                                              'kinematics', 'dip', 'dip_dir'], 
                                     crs=fault_buffers.crs)
        
        return gpd.GeoDataFrame(intersections, crs=fault_buffers.crs)
    
    def calculate_prospectivity_score(self, point, fault_data, lithology_at_point):
        """
        Calculate fuzzy prospectivity score for a point.
        
        Args:
            point: Shapely Point
            fault_data: Series with fault attributes
            lithology_at_point: Dict with lithology info at point
            
        Returns:
            float: Prospectivity score [0, 1]
        """
        # Calculate distances (simplified - in real implementation would use spatial join)
        fault_distance = 50  # Placeholder - would calculate actual distance
        contact_distance = 25  # Placeholder
        
        # Get fuzzy memberships
        mu_fault = self.fuzzy.fault_proximity(fault_distance)
        mu_contact = self.fuzzy.contact_proximity(contact_distance)
        mu_kinematic = self.fuzzy.fault_kinematic_favorability(
            fault_data['fault_type'], 
            fault_data['kinematics']
        )
        mu_host = self.fuzzy.host_rock_fertility(
            lithology_at_point.get('lithology', 'unknown'),
            lithology_at_point.get('metamorphic_grade')
        )
        
        # Apply rule weights
        weights = self.RULE_WEIGHTS
        
        # Adjust for anatectic scenario
        if self.anatectic_flag:
            mu_host = min(mu_host * 1.5, 1.0)  # Increase host rock importance
            # Granite proximity becomes less important (handled in target classification)
        
        # Fuzzy AND (minimum operator) with weighting
        raw_score = min(mu_fault, mu_contact, mu_kinematic, mu_host)
        
        # Apply combined weight factor
        weight_factor = np.mean([
            weights['fault_proximity_100m'] if mu_fault > 0.8 else 1.0,
            weights['host_rock_amphibolite_metapelite'] if mu_host > 0.8 else 1.0,
            weights['contact_proximity_50m'] if mu_contact > 0.8 else 1.0
        ])
        
        return min(raw_score * weight_factor, 1.0)
    
    def classify_target_type(self, point, fault_id, lithology_context):
        """
        Classify a point into one of the nine target types (P1-P9).
        
        Args:
            point: Shapely Point
            fault_id: Fault identifier
            lithology_context: Dict with lithology info on both sides of fault
            
        Returns:
            str: Target type code (P1-P9) or None if unclassified
        """
        # Extract fault number from ID
        try:
            fault_num = int(''.join(filter(str.isdigit, str(fault_id))))
        except:
            return None
        
        # Get lithologies on west and east sides
        west_lith = lithology_context.get('west', '').lower()
        east_lith = lithology_context.get('east', '').lower()
        
        # Classification logic based on fault position and lithologies
        if fault_num <= 2:  # Western domain
            if 'sedimentary' in west_lith and 'sedimentary' in east_lith:
                return 'P1'
            elif 'sedimentary' in west_lith and 'metamorphic' in east_lith:
                return 'P2'
            elif 'metamorphic' in west_lith and 'metamorphic' in east_lith:
                return 'P3'
            elif 'metamorphic' in west_lith and 'granite' in east_lith:
                return 'P4'
                
        elif 3 <= fault_num <= 4:  # Central/west
            if 'metamorphic' in west_lith and 'granite' in east_lith:
                return 'P4'
            elif 'granite' in west_lith and 'granite' in east_lith:
                return 'P5'
                
        elif 5 <= fault_num <= 6:  # Central/east
            if 'granite' in west_lith and 'granite' in east_lith:
                return 'P5'
            elif 'granite' in west_lith and 'metamorphic' in east_lith:
                return 'P6'
                
        elif fault_num >= 7:  # Eastern domain
            if 'metamorphic' in west_lith and 'metamorphic' in east_lith:
                return 'P7'
            elif 'metamorphic' in west_lith and 'sedimentary' in east_lith:
                return 'P8'
            elif 'sedimentary' in west_lith and 'sedimentary' in east_lith:
                return 'P9'
        
        return None
    
    def generate_targets(self, grid_spacing=100):
        """
        Generate exploration targets by evaluating the entire model area.
        
        Args:
            grid_spacing: Spacing for evaluation grid in meters
            
        Returns:
            GeoDataFrame of ranked exploration targets
        """
        if self.lithology_data is None or self.fault_data is None:
            raise ValueError("Both lithology and fault data must be loaded")
        
        # Get total bounds
        total_bounds = self.lithology_data.total_bounds
        x_min, y_min, x_max, y_max = total_bounds
        
        # Create evaluation grid
        x_coords = np.arange(x_min, x_max, grid_spacing)
        y_coords = np.arange(y_min, y_max, grid_spacing)
        
        targets = []
        
        print(f"Evaluating grid: {len(x_coords)} x {len(y_coords)} = {len(x_coords)*len(y_coords)} points")
        
        # For each point in grid
        for i, x in enumerate(x_coords):
            if i % 10 == 0:
                print(f"Processing row {i+1}/{len(x_coords)}")
                
            for y in y_coords:
                point = Point(x, y)
                
                # Find containing lithology
                containing = self.lithology_data[self.lithology_data.contains(point)]
                if len(containing) == 0:
                    continue
                
                lith_at_point = containing.iloc[0]
                
                # Find nearest fault (simplified - would use spatial index in production)
                min_fault_dist = float('inf')
                nearest_fault = None
                
                for f_idx, fault in self.fault_data.iterrows():
                    dist = point.distance(fault.geometry)
                    if dist < min_fault_dist:
                        min_fault_dist = dist
                        nearest_fault = fault
                
                if nearest_fault is None or min_fault_dist > 500:
                    continue
                
                # Calculate prospectivity
                lith_context = {
                    'lithology': lith_at_point['lithology'],
                    'metamorphic_grade': lith_at_point.get('metamorphic_grade', 'unknown')
                }
                
                score = self.calculate_prospectivity_score(point, nearest_fault, lith_context)
                
                if score > 0.3:  # Minimum threshold
                    # Determine target type (simplified)
                    fault_num = int(''.join(filter(str.isdigit, str(nearest_fault['fault_id']))))
                    
                    # Simple classification based on fault position
                    if fault_num <= 2:
                        target_type = f'P{fault_num}'
                    elif fault_num <= 4:
                        target_type = f'P{fault_num+1}'  # Adjust for P4-P5
                    elif fault_num <= 6:
                        target_type = f'P{fault_num}'  # P5-P6
                    else:
                        target_type = f'P{fault_num-2}'  # Adjust for P7-P9
                    
                    # Ensure valid target type
                    if target_type not in self.TARGET_TYPES:
                        continue
                    
                    targets.append({
                        'geometry': point,
                        'target_type': target_type,
                        'prospectivity_score': score,
                        'fault_id': nearest_fault['fault_id'],
                        'fault_distance': min_fault_dist,
                        'lithology': lith_at_point['lithology'],
                        'base_probability': self.TARGET_TYPES[target_type]['base_probability']
                    })
        
        print(f"Generated {len(targets)} potential targets")
        
        if not targets:
            return gpd.GeoDataFrame(columns=['geometry', 'target_type', 'prospectivity_score',
                                              'fault_id', 'fault_distance', 'lithology',
                                              'base_probability'], 
                                     crs=self.lithology_data.crs)
        
        targets_gdf = gpd.GeoDataFrame(targets, crs=self.lithology_data.crs)
        
        # Add priority ranking
        targets_gdf['priority'] = pd.cut(
            targets_gdf['prospectivity_score'],
            bins=[0, 0.3, 0.5, 0.7, 1.0],
            labels=['Priority 4', 'Priority 3', 'Priority 2', 'Priority 1'],
            include_lowest=True
        )
        
        # Sort by score
        targets_gdf = targets_gdf.sort_values('prospectivity_score', ascending=False)
        
        return targets_gdf
    
    def calculate_3d_volume(self, target_point, fault_dip=60, fault_dip_dir=90, plunge_angle=30):
        """
        Calculate 3D exploration volume for a target.
        
        Args:
            target_point: Shapely Point of target
            fault_dip: Fault dip angle in degrees
            fault_dip_dir: Fault dip direction in degrees
            plunge_angle: Calculated plunge angle in degrees
            
        Returns:
            dict: 3D volume parameters
        """
        x, y = target_point.x, target_point.y
        
        # Calculate plunge direction components
        plunge_rad = np.radians(plunge_angle)
        dip_rad = np.radians(fault_dip)
        
        # Plunge vector
        plunge_x = np.sin(plunge_rad) * np.cos(np.radians(fault_dip_dir))
        plunge_y = np.sin(plunge_rad) * np.sin(np.radians(fault_dip_dir))
        plunge_z = np.cos(plunge_rad)
        
        # Define volume (simplified ellipsoid)
        volume_params = {
            'center': (x, y, 0),
            'horizontal_radius': 250,  # meters
            'depth_range': (0, 3000),   # surface to 3 km depth
            'plunge_vector': (plunge_x, plunge_y, plunge_z),
            'plunge_angle': plunge_angle,
            'volume_m3': np.pi * 250**2 * 3000  # cylindrical approximation
        }
        
        return volume_params
    
    def validate_against_known(self, known_pegmatites_gdf, targets_gdf, threshold=0.5):
        """
        Validate model predictions against known pegmatite occurrences.
        
        Args:
            known_pegmatites_gdf: GeoDataFrame of known pegmatite locations
            targets_gdf: GeoDataFrame of predicted targets
            threshold: Prospectivity threshold for positive prediction
            
        Returns:
            dict: Validation metrics
        """
        # Create buffer around known pegmatites for matching
        known_buffered = known_pegmatites_gdf.copy()
        known_buffered['geometry'] = known_buffered.buffer(100)  # 100m match radius
        
        # Find hits
        hits = []
        false_positives = []
        
        for idx, target in targets_gdf.iterrows():
            if target['prospectivity_score'] >= threshold:
                # Check if within 100m of any known pegmatite
                is_hit = False
                for k_idx, known in known_buffered.iterrows():
                    if target.geometry.within(known.geometry):
                        is_hit = True
                        hits.append(target)
                        break
                
                if not is_hit:
                    false_positives.append(target)
        
        # Calculate metrics
        n_known = len(known_pegmatites_gdf)
        n_predicted_positive = len([t for t in targets_gdf.itertuples() 
                                    if t.prospectivity_score >= threshold])
        
        hit_rate = len(hits) / n_known if n_known > 0 else 0
        false_positive_rate = len(false_positives) / n_predicted_positive if n_predicted_positive > 0 else 0
        
        # Calculate ROC points for different thresholds
        thresholds = np.arange(0, 1.05, 0.05)
        tpr = []
        fpr = []
        
        for t in thresholds:
            pred_pos = [tgt for tgt in targets_gdf.itertuples() if tgt.prospectivity_score >= t]
            hits_at_t = 0
            
            for p in pred_pos:
                for k_idx, known in known_buffered.iterrows():
                    if p.geometry.within(known.geometry):
                        hits_at_t += 1
                        break
            
            tpr.append(hits_at_t / n_known if n_known > 0 else 0)
            fpr.append(len(pred_pos) / len(targets_gdf) if len(targets_gdf) > 0 else 0)
        
        # Calculate AUC using trapezoidal rule
        auc = np.trapz(tpr, fpr)
        
        return {
            'n_known': n_known,
            'n_predicted_total': len(targets_gdf),
            'n_predicted_positive': n_predicted_positive,
            'hits': len(hits),
            'false_positives': len(false_positives),
            'hit_rate': hit_rate,
            'false_positive_rate': false_positive_rate,
            'roc_curve': {'tpr': tpr, 'fpr': fpr, 'thresholds': thresholds},
            'auc': auc
        }
    
    def integrate_geochemical_data(self, targets_gdf, geochem_points_gdf, 
                                   element_thresholds=None):
        """
        Integrate geochemical data to refine target rankings.
        
        Args:
            targets_gdf: GeoDataFrame of predicted targets
            geochem_points_gdf: GeoDataFrame of geochemical samples
            element_thresholds: Dict of element thresholds for anomalies
            
        Returns:
            GeoDataFrame with updated scores
        """
        if element_thresholds is None:
            element_thresholds = {
                'Li': 200,  # ppm
                'Cs': 30,   # ppm
                'Rb': 300,  # ppm
                'Sn': 15    # ppm
            }
        
        # Spatial join to associate geochem points with targets
        joined = gpd.sjoin_nearest(targets_gdf, geochem_points_gdf, 
                                   distance_col='geochem_distance')
        
        # Calculate geochemical score for each target
        geochem_scores = {}
        
        for idx, row in joined.iterrows():
            target_idx = row.index_right if 'index_right' in joined.columns else idx
            
            score = 0
            n_elements = 0
            
            for element, threshold in element_thresholds.items():
                if element in geochem_points_gdf.columns:
                    value = row.get(element, 0)
                    if value >= threshold:
                        score += 1
                    n_elements += 1
            
            if n_elements > 0:
                geochem_score = score / n_elements
                # Apply distance weighting
                distance_weight = max(0, 1 - (row['geochem_distance'] / 1000))
                geochem_scores[target_idx] = geochem_score * distance_weight
        
        # Update targets with geochemical score
        targets_gdf = targets_gdf.copy()
        targets_gdf['geochem_score'] = targets_gdf.index.map(
            lambda x: geochem_scores.get(x, 0)
        )
        
        # Calculate final score (weighted combination)
        targets_gdf['final_score'] = (
            0.40 * targets_gdf['prospectivity_score'] +
            0.25 * targets_gdf['geochem_score']
            # Additional scores would be added here
        )
        
        return targets_gdf


# ============================================================================
# PART 2: IDEALIZED MODEL GENERATION (FOR FIGURE 1)
# ============================================================================

def create_idealized_model():
    """
    Create the idealized geological model shown in Figure 1 of the paper.
    
    Returns:
        tuple: (lithology_gdf, fault_gdf, targets_gdf)
    """
    # Create coordinate system
    x0, x10 = 0, 10000
    y0, y2 = 0, 5000
    
    # Define lithology boundaries
    lithology_polygons = []
    
    # Western sedimentary (L1)
    l1_west = box(x0, y0, 2000, y2)
    lithology_polygons.append({
        'geometry': l1_west,
        'lithology': 'sedimentary',
        'unit': 'L1',
        'metamorphic_grade': 'none'
    })
    
    # Western metamorphic (L2)
    l2_west = box(2000, y0, 4000, y2)
    lithology_polygons.append({
        'geometry': l2_west,
        'lithology': 'metamorphic',
        'unit': 'L2',
        'metamorphic_grade': 'amphibolite'
    })
    
    # Granite (L3)
    l3 = box(4000, y0, 6000, y2)
    lithology_polygons.append({
        'geometry': l3,
        'lithology': 'granite',
        'unit': 'L3',
        'metamorphic_grade': 'none'
    })
    
    # Eastern metamorphic (L2)
    l2_east = box(6000, y0, 8000, y2)
    lithology_polygons.append({
        'geometry': l2_east,
        'lithology': 'metamorphic',
        'unit': 'L2',
        'metamorphic_grade': 'amphibolite'
    })
    
    # Eastern sedimentary (L1)
    l1_east = box(8000, y0, x10, y2)
    lithology_polygons.append({
        'geometry': l1_east,
        'lithology': 'sedimentary',
        'unit': 'L1',
        'metamorphic_grade': 'none'
    })
    
    lithology_gdf = gpd.GeoDataFrame(lithology_polygons, crs='EPSG:3857')
    
    # Create faults (F1-F9)
    fault_positions = np.linspace(1000, 9000, 9)
    fault_lines = []
    
    for i, x in enumerate(fault_positions):
        fault_type = ['strike-slip', 'normal', 'reverse'][i % 3]
        kinematics = ['dilational', 'neutral', 'contractional'][i % 3]
        
        fault_lines.append({
            'geometry': LineString([(x, y0), (x, y2)]),
            'fault_id': f'F{i+1}',
            'fault_type': fault_type,
            'kinematics': kinematics,
            'dip': 75,
            'dip_dir': 90 if i < 4 else 270
        })
    
    fault_gdf = gpd.GeoDataFrame(fault_lines, crs='EPSG:3857')
    
    # Create predicted pegmatite targets (P1-P9)
    target_points = []
    
    # P1 at F1 in L1 (west)
    target_points.append({
        'geometry': Point(1000, 2500),
        'target_type': 'P1',
        'fault_id': 'F1',
        'lithology': 'sedimentary'
    })
    
    # P2 at F2 on L1-L2 contact
    target_points.append({
        'geometry': Point(2000, 2500),
        'target_type': 'P2',
        'fault_id': 'F2',
        'lithology': 'contact'
    })
    
    # P3 at F3 in L2 (west)
    target_points.append({
        'geometry': Point(3000, 2500),
        'target_type': 'P3',
        'fault_id': 'F3',
        'lithology': 'metamorphic'
    })
    
    # P4 at F4 on L2-L3 contact (west)
    target_points.append({
        'geometry': Point(4000, 2500),
        'target_type': 'P4',
        'fault_id': 'F4',
        'lithology': 'contact'
    })
    
    # P5 at F5 in L3
    target_points.append({
        'geometry': Point(5000, 2500),
        'target_type': 'P5',
        'fault_id': 'F5',
        'lithology': 'granite'
    })
    
    # P6 at F6 on L3-L2 contact (east)
    target_points.append({
        'geometry': Point(6000, 2500),
        'target_type': 'P6',
        'fault_id': 'F6',
        'lithology': 'contact'
    })
    
    # P7 at F7 in L2 (east)
    target_points.append({
        'geometry': Point(7000, 2500),
        'target_type': 'P7',
        'fault_id': 'F7',
        'lithology': 'metamorphic'
    })
    
    # P8 at F8 on L2-L1 contact (east)
    target_points.append({
        'geometry': Point(8000, 2500),
        'target_type': 'P8',
        'fault_id': 'F8',
        'lithology': 'contact'
    })
    
    # P9 at F9 in L1 (east)
    target_points.append({
        'geometry': Point(9000, 2500),
        'target_type': 'P9',
        'fault_id': 'F9',
        'lithology': 'sedimentary'
    })
    
    targets_gdf = gpd.GeoDataFrame(target_points, crs='EPSG:3857')
    
    return lithology_gdf, fault_gdf, targets_gdf


# ============================================================================
# PART 3: ZULU VALIDATION DATA (SIMPLIFIED FOR DEMONSTRATION)
# ============================================================================

def create_zulu_validation_data():
    """
    Create simplified Zulu pegmatite field data for validation.
    
    Returns:
        tuple: (lithology_gdf, fault
