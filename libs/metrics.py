import numpy as np
from scipy.stats import pearsonr
from collections import defaultdict
from shapely.geometry import Point, Polygon

def compute_clarke(pred, ref, xmin=0, xmax=400, ymin=0, ymax=400):
    # check the region
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

    results = defaultdict(int)
    for point in zip(ref, pred):
        point = Point(point)
        if region_a_1.covers(point):
            results['region_a_1'] += 1
        elif region_a_2.covers(point):
            results['region_a_2'] += 1
        elif region_b_1.covers(point):
            results['region_b_1'] += 1
        elif region_b_2.covers(point):
            results['region_b_2'] += 1
        elif region_c_1.covers(point):
            results['region_c_1'] += 1
        elif region_c_2.covers(point):
            results['region_c_2'] += 1
        elif region_d.covers(point):
            results['region_d'] += 1
        elif region_e_1.covers(point):
            results['region_e_1'] += 1
        elif region_e_2.covers(point):
            results['region_e_2'] += 1
        else:
            if point.x < xmin or point.x > xmax or point.y < ymin or point.y > ymax:
                results['out_of_bounds'] += 1

    total = sum(results.values())
    zone_a = (results['region_a_1'] + results['region_a_2'])/total
    zone_b = (results['region_b_1'] + results['region_b_2'])/total
    zone_c = (results['region_c_1'] + results['region_c_2'])/total
    zone_d = (results['region_d'])/total
    zone_e = (results['region_e_1'] + results['region_e_2'])/total
    out_of_bounds = results['out_of_bounds']/total

    corr, _ = pearsonr(pred, ref)
    return {
        'zone_a': zone_a,
        'zone_b': zone_b,
        'zone_c': zone_c,
        'zone_d': zone_d,
        'zone_e': zone_e,
        'out_of_bounds': out_of_bounds,
        'corr': corr,
    }, {
        'region_a_1': results['region_a_1'],
        'region_a_2': results['region_a_2'],
        'region_b_1': results['region_b_1'],
        'region_b_2': results['region_b_2'],
        'region_c_1': results['region_c_1'],
        'region_c_2': results['region_c_2'],
        'region_d': results['region_d'],
        'region_e_1': results['region_e_1'],
        'region_e_2': results['region_e_2'],
        'out_of_bounds': results['out_of_bounds'],
    }

def compute_hypo_metric(pred, ref, threshold=70):
    pred = np.where(pred < threshold, 1, 0)
    ref = np.where(ref < threshold, 1, 0)

    tp = np.sum((pred == 1) & (ref == 1))
    tn = np.sum((pred == 0) & (ref == 0))
    fp = np.sum((pred == 1) & (ref == 0))
    fn = np.sum((pred == 0) & (ref == 1))

    confusion_matrix = np.array([[tp, fp], [fn, tn]]) / (tp + tn + fp + fn)
    accuracy = confusion_matrix[0, 0] + confusion_matrix[1, 1]
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    return confusion_matrix, {
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
    }

def compute_rmse(pred, ref):
    return np.sqrt(np.mean((pred - ref) ** 2))