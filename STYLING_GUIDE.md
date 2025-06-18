# 🎨 Jupyter Notebook Styling Guide

This guide helps you achieve professional-quality charts and visualizations in your local Jupyter environment, matching or exceeding the quality you see in Google Colab.

## 🚀 Quick Setup

### Option 1: Automatic Setup (Recommended)

Add this cell at the beginning of any notebook:

```python
# =============================================================================
# NOTEBOOK STYLING SETUP - Run this cell first!
# =============================================================================

from jupyter_styling import setup_notebook_styling

# Setup professional styling
styling_config = setup_notebook_styling(
    style='professional',  # Options: 'professional', 'dark', 'minimal', 'colorful'
    dpi=150,              # Higher DPI for crisp charts (100 for standard, 150+ for high-DPI)
    enable_plotly=True    # Enable plotly configuration
)

print("🎨 Notebook styling configured successfully!")
```

### Option 2: Copy-Paste Template

Copy the contents of `notebook_setup_cell.py` into your first notebook cell.

## 🎯 Styling Options

### Professional Style (Default)
- Clean, publication-ready appearance
- Subtle grids and professional color palette
- Optimized for business presentations

```python
setup_notebook_styling(style='professional')
```

### Dark Theme
- Dark background with bright colors
- Great for presentations and reducing eye strain
- Modern, sleek appearance

```python
setup_notebook_styling(style='dark')
```

### Minimal Style
- Clean, minimal design
- No grids, simple lines
- Perfect for academic papers

```python
setup_notebook_styling(style='minimal')
```

### Colorful Style
- Vibrant colors and engaging visuals
- Great for educational content
- Eye-catching presentations

```python
setup_notebook_styling(style='colorful')
```

## 📊 Chart Quality Settings

### High-DPI Displays
For crisp charts on high-resolution displays:

```python
setup_notebook_styling(dpi=200)  # Very high quality
setup_notebook_styling(dpi=150)  # High quality (recommended)
setup_notebook_styling(dpi=100)  # Standard quality
```

### Figure Sizes
Charts are automatically sized for optimal display, but you can customize:

```python
import matplotlib.pyplot as plt

# For wide charts
plt.figure(figsize=(12, 6))

# For square charts
plt.figure(figsize=(8, 8))

# For tall charts
plt.figure(figsize=(8, 10))
```

## 🎨 Color Palettes

### Matplotlib/Seaborn
The styling system automatically configures professional color palettes:

```python
# These are automatically set by the styling system
sns.set_palette("husl")      # Professional style
sns.set_palette("bright")    # Dark style
sns.set_palette("muted")     # Minimal style
sns.set_palette("Set2")      # Colorful style
```

### Plotly
Interactive charts use optimized themes:

```python
import plotly.io as pio

# These are automatically configured
pio.templates.default = "plotly_white"  # Professional
pio.templates.default = "plotly_dark"   # Dark
pio.templates.default = "simple_white"  # Minimal
pio.templates.default = "plotly"        # Colorful
```

## 📈 Chart Examples

### Enhanced Bar Chart
```python
import matplotlib.pyplot as plt
import numpy as np

# Data
tools = ['Pandas', 'Polars', 'DuckDB', 'Spark']
performance = [1.2, 0.8, 0.6, 2.1]

# Create chart
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(tools, performance, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{height:.1f}s', ha='center', va='bottom', fontweight='bold')

# Styling
ax.set_title('Performance Comparison', fontsize=16, fontweight='bold', pad=20)
ax.set_ylabel('Execution Time (seconds)', fontsize=12)
ax.set_xlabel('Data Processing Tools', fontsize=12)

# Remove top and right spines (automatically done by styling)
plt.tight_layout()
plt.show()
```

### Interactive Plotly Chart
```python
import plotly.graph_objects as go

# Data
tools = ['Pandas', 'Polars', 'DuckDB', 'Spark']
performance = [1.2, 0.8, 0.6, 2.1]

# Create interactive chart
fig = go.Figure(data=[
    go.Bar(
        x=tools, 
        y=performance,
        text=[f'{v:.1f}s' for v in performance],
        textposition='auto',
        marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    )
])

fig.update_layout(
    title='Interactive Performance Comparison',
    xaxis_title='Data Processing Tools',
    yaxis_title='Execution Time (seconds)',
    font=dict(size=12),
    height=500,
    margin=dict(l=50, r=50, t=80, b=50)
)

fig.show()
```

### Seaborn Statistical Plot
```python
import seaborn as sns
import pandas as pd

# Sample data
data = pd.DataFrame({
    'Tool': ['Pandas', 'Polars', 'DuckDB', 'Spark'] * 10,
    'Performance': np.random.normal([1.2, 0.8, 0.6, 2.1] * 10, 0.1),
    'Dataset_Size': ['Small', 'Medium', 'Large', 'XLarge'] * 10
})

# Create plot
plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x='Tool', y='Performance', hue='Dataset_Size')
plt.title('Performance Distribution by Tool and Dataset Size', fontsize=14, fontweight='bold')
plt.ylabel('Execution Time (seconds)')
plt.tight_layout()
plt.show()
```

## 🛠️ Troubleshooting

### Charts Look Blurry
```python
# Increase DPI
setup_notebook_styling(dpi=200)

# Or set matplotlib DPI directly
plt.rcParams['figure.dpi'] = 200
plt.rcParams['savefig.dpi'] = 200
```

### Plotly Charts Not Interactive
```python
# Ensure proper renderer
import plotly.io as pio
pio.renderers.default = "notebook"

# For JupyterLab
pio.renderers.default = "jupyterlab"
```

### Fonts Look Different
```python
# Set specific font family
plt.rcParams['font.family'] = 'Arial'  # or 'DejaVu Sans', 'Liberation Sans'
```

### Charts Too Small/Large
```python
# Adjust default figure size
plt.rcParams['figure.figsize'] = (12, 8)  # width, height in inches
```

## 📱 Responsive Design

### For Different Screen Sizes
```python
# Laptop/Desktop
setup_notebook_styling(dpi=150)
plt.rcParams['figure.figsize'] = (10, 6)

# Large Monitor
setup_notebook_styling(dpi=200)
plt.rcParams['figure.figsize'] = (12, 8)

# Presentation Mode
setup_notebook_styling(dpi=150)
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 14
```

## 🎯 Performance Metrics Display

### Styled Performance Badges
```python
from jupyter_styling import create_performance_badge
from IPython.display import HTML, display

# Create performance indicators
time_badge = create_performance_badge(1.2, 'time')
memory_badge = create_performance_badge(256, 'memory')
throughput_badge = create_performance_badge(1500, 'throughput')

display(HTML(f"""
<div>
    <h4>Performance Metrics:</h4>
    <p>Execution Time: {time_badge}</p>
    <p>Memory Usage: {memory_badge} MB</p>
    <p>Throughput: {throughput_badge} rows/sec</p>
</div>
"""))
```

## 🔧 Advanced Customization

### Custom Color Schemes
```python
# Define custom colors
custom_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

# Apply to matplotlib
plt.rcParams['axes.prop_cycle'] = plt.cycler('color', custom_colors)

# Apply to seaborn
sns.set_palette(custom_colors)
```

### Custom Plotly Theme
```python
import plotly.graph_objects as go

# Create custom template
custom_template = go.layout.Template(
    layout=go.Layout(
        font=dict(family="Arial", size=12),
        colorway=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'],
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
)

# Apply template
pio.templates['custom'] = custom_template
pio.templates.default = 'custom'
```

## 📋 Best Practices

### 1. Always Run Styling First
- Place styling setup in the first cell
- Run before any plotting code

### 2. Consistent Sizing
- Use consistent figure sizes across notebooks
- Consider your target display/output format

### 3. Color Accessibility
- Test charts with colorblind-friendly palettes
- Ensure sufficient contrast

### 4. Export Quality
```python
# For high-quality exports
plt.savefig('chart.png', dpi=300, bbox_inches='tight', facecolor='white')
```

### 5. Interactive vs Static
- Use Plotly for interactive exploration
- Use Matplotlib/Seaborn for static reports

## 🆘 Getting Help

If you encounter issues:

1. **Check Dependencies**: Run `python verify_setup.py`
2. **Restart Kernel**: Sometimes required after styling changes
3. **Clear Output**: Clear all outputs and re-run cells
4. **Check Browser**: Some features require modern browsers

## 📚 Additional Resources

- [Matplotlib Styling Guide](https://matplotlib.org/stable/tutorials/introductory/customizing.html)
- [Seaborn Tutorial](https://seaborn.pydata.org/tutorial.html)
- [Plotly Documentation](https://plotly.com/python/)

---

🎉 **Happy Plotting!** Your charts will now look professional and consistent across all notebooks in the DataProcessingComparison project.