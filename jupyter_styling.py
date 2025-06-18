#!/usr/bin/env python3
"""
Jupyter Notebook Styling for DataProcessingComparison
=====================================================

This module provides comprehensive styling configuration for Jupyter notebooks
to ensure charts and visualizations look professional and consistent across
different environments (local Jupyter, JupyterLab, etc.).

Usage:
    # At the beginning of any notebook, add:
    from jupyter_styling import setup_notebook_styling
    setup_notebook_styling()

Features:
- High-DPI display support
- Professional color schemes
- Consistent font styling
- Optimized chart layouts
- Interactive plot configurations
- Dark/light theme support
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from IPython.display import HTML, display
import warnings

# Suppress common warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

def setup_matplotlib_styling(style='professional', dpi=100):
    """
    Configure matplotlib for professional-looking charts.
    
    Args:
        style: 'professional', 'dark', 'minimal', or 'colorful'
        dpi: Display resolution (100 for standard, 150+ for high-DPI)
    """
    # Set high-quality rendering
    plt.rcParams['figure.dpi'] = dpi
    plt.rcParams['savefig.dpi'] = dpi
    
    # Font configuration
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    
    # Layout and spacing
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['figure.autolayout'] = True
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    
    # Style-specific configurations
    if style == 'professional':
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['axes.facecolor'] = '#f8f9fa'
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['grid.color'] = '#e9ecef'
        
    elif style == 'dark':
        plt.style.use('dark_background')
        plt.rcParams['axes.facecolor'] = '#2f3136'
        plt.rcParams['figure.facecolor'] = '#36393f'
        plt.rcParams['text.color'] = 'white'
        plt.rcParams['axes.labelcolor'] = 'white'
        plt.rcParams['xtick.color'] = 'white'
        plt.rcParams['ytick.color'] = 'white'
        
    elif style == 'minimal':
        plt.style.use('seaborn-v0_8-white')
        plt.rcParams['axes.spines.left'] = True
        plt.rcParams['axes.spines.bottom'] = True
        plt.rcParams['axes.grid'] = False
        
    elif style == 'colorful':
        plt.style.use('seaborn-v0_8-bright')
        plt.rcParams['axes.prop_cycle'] = plt.cycler('color', 
            ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
             '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])

def setup_seaborn_styling(style='professional', palette='husl'):
    """
    Configure seaborn for consistent styling.
    
    Args:
        style: 'professional', 'dark', 'minimal', or 'colorful'
        palette: Color palette name
    """
    if style == 'professional':
        sns.set_style("whitegrid")
        sns.set_palette("husl")
        
    elif style == 'dark':
        sns.set_style("darkgrid")
        sns.set_palette("bright")
        
    elif style == 'minimal':
        sns.set_style("white")
        sns.set_palette("muted")
        
    elif style == 'colorful':
        sns.set_style("whitegrid")
        sns.set_palette("Set2")
    
    # Common seaborn settings
    sns.set_context("notebook", font_scale=1.1)

def setup_plotly_styling(theme='professional'):
    """
    Configure plotly for interactive charts.
    
    Args:
        theme: 'professional', 'dark', 'minimal', or 'colorful'
    """
    if theme == 'professional':
        pio.templates.default = "plotly_white"
        
    elif theme == 'dark':
        pio.templates.default = "plotly_dark"
        
    elif theme == 'minimal':
        pio.templates.default = "simple_white"
        
    elif theme == 'colorful':
        pio.templates.default = "plotly"
    
    # Global plotly configuration
    config = {
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'chart',
            'height': 600,
            'width': 1000,
            'scale': 2
        }
    }
    
    # Set default config
    pio.renderers.default = "notebook"
    
    return config

def setup_jupyter_display():
    """
    Configure Jupyter display settings for better chart rendering.
    """
    # Enable inline plotting
    from IPython import get_ipython
    ipython = get_ipython()
    if ipython:
        ipython.magic('matplotlib inline')
    
    # Custom CSS for better notebook appearance
    css = """
    <style>
    /* Improve chart containers */
    .output_png {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    
    /* Better spacing for plots */
    .output_subarea {
        max-width: 100%;
        margin: 10px 0;
    }
    
    /* Improve plotly chart rendering */
    .plotly-graph-div {
        margin: 10px auto;
        border: 1px solid #e1e4e8;
        border-radius: 6px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* Better table styling */
    .dataframe {
        border: 1px solid #e1e4e8;
        border-radius: 6px;
        overflow: hidden;
        margin: 10px 0;
    }
    
    .dataframe th {
        background-color: #f6f8fa;
        font-weight: 600;
        text-align: left;
        padding: 8px 12px;
    }
    
    .dataframe td {
        padding: 8px 12px;
        border-top: 1px solid #e1e4e8;
    }
    
    /* Improve code cell appearance */
    .CodeMirror {
        font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
        font-size: 13px;
        line-height: 1.4;
    }
    
    /* Better markdown rendering */
    .text_cell_render h1, .text_cell_render h2, .text_cell_render h3 {
        color: #24292e;
        font-weight: 600;
        margin-top: 24px;
        margin-bottom: 16px;
    }
    
    .text_cell_render p {
        margin-bottom: 16px;
        line-height: 1.6;
    }
    
    /* Performance indicators styling */
    .performance-metric {
        display: inline-block;
        padding: 4px 8px;
        margin: 2px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 12px;
    }
    
    .metric-excellent { background-color: #d4edda; color: #155724; }
    .metric-good { background-color: #d1ecf1; color: #0c5460; }
    .metric-average { background-color: #fff3cd; color: #856404; }
    .metric-poor { background-color: #f8d7da; color: #721c24; }
    </style>
    """
    
    display(HTML(css))

def create_performance_badge(value, metric_type='time', thresholds=None):
    """
    Create a styled performance badge for metrics.
    
    Args:
        value: The metric value
        metric_type: 'time', 'memory', 'throughput', or 'custom'
        thresholds: Custom thresholds for classification
    
    Returns:
        HTML string for the badge
    """
    if thresholds is None:
        if metric_type == 'time':
            thresholds = {'excellent': 1.0, 'good': 5.0, 'average': 15.0}
        elif metric_type == 'memory':
            thresholds = {'excellent': 100, 'good': 500, 'average': 1000}  # MB
        elif metric_type == 'throughput':
            thresholds = {'excellent': 1000, 'good': 500, 'average': 100}  # rows/sec
        else:
            thresholds = {'excellent': 80, 'good': 60, 'average': 40}  # percentage
    
    if metric_type == 'throughput':
        if value >= thresholds['excellent']:
            css_class = 'metric-excellent'
        elif value >= thresholds['good']:
            css_class = 'metric-good'
        elif value >= thresholds['average']:
            css_class = 'metric-average'
        else:
            css_class = 'metric-poor'
    else:
        if value <= thresholds['excellent']:
            css_class = 'metric-excellent'
        elif value <= thresholds['good']:
            css_class = 'metric-good'
        elif value <= thresholds['average']:
            css_class = 'metric-average'
        else:
            css_class = 'metric-poor'
    
    return f'<span class="performance-metric {css_class}">{value}</span>'

def setup_notebook_styling(style='professional', dpi=100, enable_plotly=True):
    """
    Complete notebook styling setup.
    
    Args:
        style: Overall style theme ('professional', 'dark', 'minimal', 'colorful')
        dpi: Display resolution for matplotlib
        enable_plotly: Whether to configure plotly
    
    Returns:
        Dictionary with styling configuration
    """
    print("🎨 Setting up notebook styling...")
    
    # Setup matplotlib
    setup_matplotlib_styling(style=style, dpi=dpi)
    print(f"✅ Matplotlib configured with '{style}' style")
    
    # Setup seaborn
    setup_seaborn_styling(style=style)
    print(f"✅ Seaborn configured with '{style}' style")
    
    # Setup plotly if enabled
    plotly_config = None
    if enable_plotly:
        plotly_config = setup_plotly_styling(theme=style)
        print(f"✅ Plotly configured with '{style}' theme")
    
    # Setup Jupyter display
    setup_jupyter_display()
    print("✅ Jupyter display settings configured")
    
    print("🎉 Notebook styling setup complete!")
    
    return {
        'style': style,
        'dpi': dpi,
        'plotly_config': plotly_config,
        'matplotlib_backend': plt.get_backend()
    }

def create_comparison_chart(data, title="Performance Comparison", chart_type='bar'):
    """
    Create a standardized comparison chart.
    
    Args:
        data: Dictionary with tool names as keys and values as values
        title: Chart title
        chart_type: 'bar', 'line', or 'radar'
    
    Returns:
        Matplotlib figure or Plotly figure
    """
    if chart_type == 'bar':
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(data.keys(), data.values())
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom')
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_ylabel('Value')
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig
        
    elif chart_type == 'plotly_bar':
        fig = go.Figure(data=[
            go.Bar(x=list(data.keys()), y=list(data.values()),
                   text=[f'{v:.2f}' for v in data.values()],
                   textposition='auto')
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title="Tools",
            yaxis_title="Value",
            font=dict(size=12),
            height=500,
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        return fig

# Example usage and testing
def test_styling():
    """Test function to demonstrate styling capabilities."""
    import numpy as np
    
    print("🧪 Testing styling configuration...")
    
    # Test matplotlib
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    
    ax1.plot(x, y1, label='sin(x)', linewidth=2)
    ax1.plot(x, y2, label='cos(x)', linewidth=2)
    ax1.set_title('Matplotlib Test')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Test bar chart
    tools = ['Pandas', 'Polars', 'DuckDB', 'Spark']
    performance = [1.2, 0.8, 0.6, 2.1]
    
    bars = ax2.bar(tools, performance)
    ax2.set_title('Performance Comparison')
    ax2.set_ylabel('Time (seconds)')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Styling test completed!")

if __name__ == "__main__":
    # Demo the styling setup
    config = setup_notebook_styling()
    test_styling()