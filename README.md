# GlucoseBench

**GlucoseBench** is a repository for evaluating and visualizing the performance of glucose prediction models. It provides clinically meaningful metrics and visual tools such as the Clarke Error Grid, hypoglycemia-focused scores, distribution plots, and RMSE visualizations.

## 🔍 Key Features

- [x] **Clarke Error Grid** (point + heatmap styles)
- [x] **Hypoglycemia-focused metric**
- [x] **Correlation and RMSE**
- [x] **Prediction vs. Ground Truth distribution plots**

## 📦 Installation

```bash
git clone https://github.com/PSI-TAMU/GlucoseBench.git
cd GlucoseBench
pip install -e .
```

## 🚀 Quick Start

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glucosebench.plot import plot_clarke_error_grid, plot_hypo_metric, plot_distribution, plot_rmse

# Load predictions
df = pd.read_csv('./samples/01.csv')
pred_glucose = df['pred'].values
gt_glucose = df['gt'].values
```

### 🟦 Clarke Error Grid

```python
fig, ax = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

clarke_score = plot_clarke_error_grid(ax[0], pred_glucose, gt_glucose, style='point')
ax[0].set_title(f"Correlation: {clarke_score['corr']:.2f}")

plot_clarke_error_grid(ax[1], pred_glucose, gt_glucose, style='heatmap', bin_size=3)
ax[1].set_title(f"Correlation: {clarke_score['corr']:.2f}")
plt.show()

print(clarke_score)
```

### 🟨 Hypoglycemia Metric

```python
fig, ax = plt.subplots(figsize=(8, 6))
hypo_score = plot_hypo_metric(ax, pred_glucose, gt_glucose)
plt.show()

print(hypo_score)
```

### 📈 Distribution and RMSE Plots

```python
fig, ax = plt.subplots(figsize=(8, 6))
plot_distribution(ax, pred_glucose, gt_glucose)
plt.show()

fig, ax = plt.subplots(figsize=(8, 6))
rmse = plot_rmse(ax, pred_glucose, gt_glucose, xmin=40, xmax=180, ymin=40, ymax=180)
plt.show()

print(rmse)
```

## 📁 Directory Structure

```
GlucoseBench/
├── glucosebench/
│   ├── plot.py               # Clarke grid, RMSE, hypo metrics, etc.
├── samples/
│   └── 01.csv                # Sample prediction CSV (gt, pred)
├── demo.ipynb               # Full example notebook
├── requirements.txt
└── README.md
```

## 📄 License

MIT License © 2025 PSI-TAMU
