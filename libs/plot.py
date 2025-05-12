import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import Polygon, Point
from collections import defaultdict
from libs.metrics import compute_clarke

def plot_clarke_error_grid(ax, pred, gt, xmin=0, xmax=400, ymin=0, ymax=400, style='point', bin_size=1):
    clarke_score, clarke_score_details = compute_clarke(pred, gt, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
    # plot the grid
    ax.plot([70/1.2, ymax/1.2], [70, ymax], color='k' if style=='point' else 'white', lw=2)
    ax.plot([70, xmax], [70 * 0.8, xmax * 0.8], color='k' if style=='point' else 'white', lw=2)
    ax.plot([70, 70+(ymax-180)], [180, ymax], color='k' if style=='point' else 'white', lw=2)
    ax.plot([180-(70-ymin), 180], [ymin, 70], color='k' if style=='point' else 'white', lw=2)
    ax.hlines(70, xmin, 70/1.2, color='k' if style=='point' else 'white', lw=2)
    ax.hlines(180, xmin, 70, color='k' if style=='point' else 'white', lw=2)
    ax.hlines(70, 180, xmax, color='k' if style=='point' else 'white', lw=2)
    ax.vlines(70, ymin, ymax, color='k' if style=='point' else 'white', lw=2)
    ax.vlines(180, ymin, 70, color='k' if style=='point' else 'white', lw=2)

    region_a_1_points = [(70, 70*1.2), (ymax/1.2, ymax), (ymax, ymax), (xmax, xmax*0.8), (70, 70 * 0.8), (70, 70*0.8)]
    region_a_2_points = [(xmin, ymin), (xmin, 70), (70 / 1.2, 70), (70, 70*1.2), (70, ymin)]
    region_b_1_points = [(70, 70*1.2), (70, 180), (70+(ymax-180), ymax), (ymax/1.2, ymax)]
    region_b_2_points = [(70, ymin), (70, 70*0.8), (xmax, xmax*0.8), (xmax, 70), (180, 70), (180-(70-ymin), ymin)]
    region_c_1_points = [(70,180), (70, ymax), (70+(ymax-180), ymax)]
    region_c_2_points = [(180, ymin), (180, 70), (180-(70-ymin), ymin)]
    region_d_points = [(xmin, 70), (70/1.2, 70), (70, 70 * 1.2), (70, 180), (xmin, 180)]
    region_e_1_points = [(xmin, 180), (xmin, ymax), (70, ymax), (70, 180)]
    region_e_2_points = [(180, ymin), (180, 70), (xmax, 70), (xmax, ymin)]

    # create the polygons
    region_a_1 = Polygon(region_a_1_points)
    region_a_2 = Polygon(region_a_2_points)
    region_b_1 = Polygon(region_b_1_points)
    region_b_2 = Polygon(region_b_2_points)
    region_c_1 = Polygon(region_c_1_points)
    region_c_2 = Polygon(region_c_2_points)
    region_d = Polygon(region_d_points)
    region_e_1 = Polygon(region_e_1_points)
    region_e_2 = Polygon(region_e_2_points)

    if style == 'point':
        ax.add_patch(MplPolygon(region_a_1_points, closed=True, facecolor='lightblue', edgecolor='blue', alpha=0.5))
        ax.add_patch(MplPolygon(region_a_2_points, closed=True, facecolor='lightblue', edgecolor='blue', alpha=0.5))    
        ax.add_patch(MplPolygon(region_b_1_points, closed=True, facecolor='lightgreen', edgecolor='green', alpha=0.5))
        ax.add_patch(MplPolygon(region_b_2_points, closed=True, facecolor='lightgreen', edgecolor='green', alpha=0.5))
        ax.add_patch(MplPolygon(region_c_1_points, closed=True, facecolor='lightcoral', edgecolor='red', alpha=0.5))
        ax.add_patch(MplPolygon(region_c_2_points, closed=True, facecolor='lightcoral', edgecolor='red', alpha=0.5))
        ax.add_patch(MplPolygon(region_d_points, closed=True, facecolor='lightyellow', edgecolor='orange', alpha=0.5))
        ax.add_patch(MplPolygon(region_e_1_points, closed=True, facecolor='lightpink', edgecolor='purple', alpha=0.5))
        ax.add_patch(MplPolygon(region_e_2_points, closed=True, facecolor='lightpink', edgecolor='purple', alpha=0.5))
        ax.scatter(gt, pred, s=1, c='red', alpha=0.3)
    elif style == 'heatmap':
        bins = np.arange(xmin, xmax + bin_size, bin_size)
        H, xedges, yedges = np.histogram2d(gt, pred, bins=[bins, bins], density=True)
        mesh = ax.pcolormesh(
            xedges, yedges, H.T,
            shading='auto',
            cmap='viridis'
        )
        cbar = plt.colorbar(mesh, ax=ax, pad=0.02)
        cbar.set_label(f'Density (bin = {bin_size}×{bin_size} mg/dL)')
        cbar.ax.tick_params(labelsize=12)
    else:
        raise ValueError('style must be either point or heatmap')
    
    total = sum(clarke_score_details.values())
    for region_name, region in zip(['region_a_1', 'region_a_2', 'region_b_1', 'region_b_2', 'region_c_1', 'region_c_2', 'region_d', 'region_e_1', 'region_e_2'], 
                            [region_a_1, region_a_2, region_b_1, region_b_2, region_c_1, region_c_2, region_d, region_e_1, region_e_2]):
        ax.text(region.centroid.x, region.centroid.y, f'{clarke_score_details[region_name]/total * 100:.1f}%', ha='center', va='center', fontsize=10, color='black' if style=='point' else 'white')
    
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel('GT')
    ax.set_ylabel('Predicted')
    return clarke_score


def draw_clarke_heatmap(ax, ref, pred, 
                        xmin=30, xmax=190, ymin=30, ymax=190,
                        bin_size=1):
    """
    Plots a Clarke error grid heatmap (counts per 1×1 mg/dL bin)
    for reference vs. predicted glucose, then overlays the
    Clarke regions.
    
    ax   : matplotlib Axes
    ref  : array-like of reference glucose values
    pred : array-like of predicted glucose values
    """
    # --- 1) compute 2D histogram at 1 mg/dL resolution ---
    bins = np.arange(xmin, xmax + bin_size, bin_size)
    H, xedges, yedges = np.histogram2d(ref, pred, bins=[bins, bins], density=True)
    
    # plot heatmap
    mesh = ax.pcolormesh(
        xedges, yedges, H.T,
        shading='auto',
        cmap='viridis'
    )
    cbar = plt.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label(f'Density (bin = {bin_size}×{bin_size} mg/dL)')
    cbar.ax.tick_params(labelsize=12)
    
    # --- 2) overlay Clarke grid lines ---
    # Lines: y = 1.2x region boundaries
    ax.plot([70/1.2, ymax/1.2], [70, ymax], color='white', lw=2)
    ax.plot([70, xmax], [70*.8, xmax*.8], color='white', lw=2)
    ax.plot([70, 70+(ymax-180)], [180, ymax], color='white', lw=2)
    ax.plot([180-(70-ymin), 180], [ymin, 70], color='white', lw=2)
    ax.hlines(70, xmin, 70/1.2,   color='white', lw=2)
    ax.hlines(180, xmin, 70,      color='white', lw=2)
    ax.hlines(70, 180, xmax,      color='white', lw=2)
    ax.vlines(70, ymin, ymax,     color='white', lw=2)
    ax.vlines(180, ymin, 70,      color='white', lw=2)
    
    # --- 3) shade Clarke regions and annotate percentages ---
    # (same region definitions as your original function)
    regions = {}
    # region A1
    regions['A1'] = Polygon([
        (70, 70*1.2), (ymax/1.2, ymax), (ymax, ymax),
        (xmax, xmax*0.8), (70, 70*0.8), (70, 70*0.8)
    ])
    # region A2
    regions['A2'] = Polygon([
        (xmin, ymin), (xmin, 70), (70/1.2, 70),
        (70, 70*1.2), (70, ymin)
    ])
    # region B1
    regions['B1'] = Polygon([
        (70, 70*1.2), (70, 180), (70+(ymax-180), ymax),
        (ymax/1.2, ymax)
    ])
    # region B2
    regions['B2'] = Polygon([
        (70, ymin), (70, 70*0.8), (xmax, xmax*0.8),
        (xmax, 70), (180, 70), (180-(70-ymin), ymin)
    ])
    # region C1
    regions['C1'] = Polygon([
        (70,180), (70, ymax), (70+(ymax-180), ymax)
    ])
    # region C2
    regions['C2'] = Polygon([
        (180, ymin), (180, 70), (180-(70-ymin), ymin)
    ])
    # region D
    regions['D'] = Polygon([
        (xmin, 70), (70/1.2, 70), (70, 70*1.2),
        (70, 180), (xmin, 180)
    ])
    # region E1
    regions['E1'] = Polygon([
        (xmin, 180), (xmin, ymax), (70, ymax), (70, 180)
    ])
    # region E2
    regions['E2'] = Polygon([
        (180, ymin), (180, 70), (xmax, 70), (xmax, ymin)
    ])
    
    # count points per region
    counts = defaultdict(int)
    total = len(ref)
    for x,y in zip(ref, pred):
        p = Point(x,y)
        for name, poly in regions.items():
            if poly.covers(p):
                counts[name] += 1
                break
    
    # shade & annotate
    for name, poly in regions.items():
        cen = poly.centroid
        pct = counts[name] / total * 100
        ax.text(
            cen.x, cen.y,
            f'{pct:.1f}%',
            ha='center', va='center',
            fontsize=12, color='white',
        )
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel('Reference Glucose (mg/dL)', fontsize=14)
    ax.set_ylabel('Predicted Glucose (mg/dL)', fontsize=14)
    ax.set_title(f'Clarke Error Grid with {bin_size}×{bin_size} mg/dL Heatmap', fontsize=16)
    return mesh, counts
