import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import Polygon, Point
from collections import defaultdict
from glucosebench.metrics import compute_clarke, compute_hypo_metric, compute_rmse

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

def plot_hypo_metric(ax, pred, gt):
    cm, score = compute_hypo_metric(pred, gt, threshold=70)
    sns.heatmap(cm, annot=True, fmt=".2f", cmap='Blues', ax=ax)
    ax.set_xticklabels(['Hypo', 'Normal'])
    ax.set_yticklabels(['Hypo', 'Normal'])
    ax.set_xlabel('GT')
    ax.set_ylabel('Predicted')
    ax.set_title('Train - Accuracy: {:.2f} | Sensitivity: {:.2f} | Specificity: {:.2f}'.format(score['accuracy'], score['sensitivity'], score['specificity']))
    return score

def plot_distribution(ax, pred, gt, bins=50):
    # histogram
    ax.hist(pred, bins=bins, alpha=0.5, label='Predicted', density=True, color='blue')
    ax.hist(gt, bins=bins, alpha=0.5, label='GT', density=True, color='orange')
    ax.set_xlabel('Glucose')
    ax.set_ylabel('Density')
    ax.legend()
    return

def plot_rmse(ax, pred, gt, xmin=0, xmax=400, ymin=0, ymax=400, bin_size=5):
    rmse = compute_rmse(pred, gt)

    _dict = defaultdict(list)
    for i in range(pred.shape[0]):
        _dict[float(gt[i])].append(float(pred[i]))

    xrange_mean = []
    xrange_std = []
    for pos in range(40, 180, bin_size):
        _xrange = []
        for key in sorted(_dict.keys()):
            if key >= pos and key < pos + bin_size:
                _xrange.extend(_dict[key])
        if len(_xrange) > 0:
            xrange_std.append(np.std(np.array(_xrange)))
            xrange_mean.append(np.mean(np.array(_xrange)))
        else:
            xrange_std.append(0)
            xrange_mean.append(0)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    ax.plot(xlim, xlim, color='k', lw=2, linestyle='--')
    ax.plot(range(40, 180, bin_size), xrange_mean, label='Predicted', color='red')
    ax.hlines(70, xlim[0], xlim[1], color='blue', lw=2, linestyle='--')
    ax.vlines(70, ylim[0], ylim[1], color='blue', lw=2, linestyle='--')
    ax.fill_between(range(40, 180, bin_size), np.array(xrange_mean) - np.array(xrange_std), np.array(xrange_mean) + np.array(xrange_std), alpha=0.2, color='red')
    ax.set_xlabel('GT')
    ax.set_ylabel('Predicted')

    ax.set_title('RMSE: {:.2f}'.format(rmse))
    ax.legend()
    return rmse


def plot_summary(pred_glucose, gt_glucose, xmin=0, xmax=400, ymin=0, ymax=400):
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    clarke_score = plot_clarke_error_grid(ax[0][0], pred_glucose, gt_glucose, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, style='point')
    ax[0][0].set_title(f"Correlation: {clarke_score['corr']:.2f}")
    hypo_score = plot_hypo_metric(ax[0][1], pred_glucose, gt_glucose)
    plot_distribution(ax[1][0], pred_glucose, gt_glucose)
    ax[1][0].set_title('Distribution of Glucose')
    rmse = plot_rmse(ax[1][1], pred_glucose, gt_glucose)
    plt.show()
    return clarke_score, hypo_score, rmse