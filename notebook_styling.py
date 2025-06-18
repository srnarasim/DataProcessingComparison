"""
Jupyter Notebook Styling and Visualization Enhancement
=====================================================

This module provides comprehensive styling and visualization enhancements
for Jupyter notebooks to match Google Colab's polished appearance.

Features:
- Enhanced Plotly chart rendering and styling
- Improved matplotlib themes and sizing
- Custom CSS styling for notebooks
- High-DPI display support
- Interactive chart configurations
- Consistent color schemes across all visualizations
"""

import warnings
warnings.filterwarnings('ignore')

def setup_notebook_styling():
    """
    Set up comprehensive notebook styling for better chart display
    """
    print("🎨 Setting up enhanced notebook styling...")
    
    # Import required libraries
    try:
        from IPython.display import HTML, display
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        
        # Configure matplotlib for better display
        setup_matplotlib_styling()
        
        # Set up Plotly configuration
        setup_plotly_styling()
        
        # Apply custom CSS styling
        apply_custom_css()
        
        print("✅ Notebook styling configured successfully!")
        print("📊 Charts will now display with enhanced styling")
        
    except ImportError as e:
        print(f"⚠️ Some styling features unavailable: {e}")

def setup_matplotlib_styling():
    """Configure matplotlib for better chart appearance"""
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    
    # Set high-DPI display support
    try:
        from IPython.display import set_matplotlib_formats
        set_matplotlib_formats('retina', 'png')
    except:
        pass
    
    # Configure matplotlib parameters
    plt.rcParams.update({
        # Figure settings
        'figure.figsize': (12, 8),
        'figure.dpi': 100,
        'figure.facecolor': 'white',
        'figure.edgecolor': 'none',
        
        # Axes settings
        'axes.facecolor': 'white',
        'axes.edgecolor': '#CCCCCC',
        'axes.linewidth': 1,
        'axes.grid': True,
        'axes.axisbelow': True,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'axes.labelcolor': '#333333',
        'axes.spines.top': False,
        'axes.spines.right': False,
        
        # Grid settings
        'grid.color': '#E0E0E0',
        'grid.linestyle': '-',
        'grid.linewidth': 0.5,
        'grid.alpha': 0.7,
        
        # Font settings
        'font.family': ['DejaVu Sans', 'Arial', 'sans-serif'],
        'font.size': 11,
        'font.weight': 'normal',
        
        # Legend settings
        'legend.frameon': True,
        'legend.fancybox': True,
        'legend.shadow': False,
        'legend.framealpha': 0.9,
        'legend.facecolor': 'white',
        'legend.edgecolor': '#CCCCCC',
        
        # Tick settings
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'xtick.color': '#666666',
        'ytick.color': '#666666',
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        
        # Line settings
        'lines.linewidth': 2,
        'lines.markersize': 6,
        
        # Savefig settings
        'savefig.dpi': 150,
        'savefig.bbox': 'tight',
        'savefig.facecolor': 'white',
        'savefig.edgecolor': 'none'
    })
    
    # Set a modern color palette
    modern_colors = [
        '#1f77b4',  # Blue
        '#ff7f0e',  # Orange  
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd',  # Purple
        '#8c564b',  # Brown
        '#e377c2',  # Pink
        '#7f7f7f',  # Gray
        '#bcbd22',  # Olive
        '#17becf'   # Cyan
    ]
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=modern_colors)

def setup_plotly_styling():
    """Configure Plotly for better chart appearance in Jupyter"""
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
        from plotly.offline import init_notebook_mode
        
        # Initialize Plotly for Jupyter
        init_notebook_mode(connected=True)
        
        # Set default renderer for Jupyter
        pio.renderers.default = "notebook"
        
        # Configure default template
        pio.templates.default = "plotly_white"
        
        # Create custom template for better appearance
        custom_template = go.layout.Template(
            layout=go.Layout(
                # Overall layout
                font=dict(family="Arial, sans-serif", size=12, color="#333333"),
                title=dict(font=dict(size=16, color="#333333"), x=0.5, xanchor='center'),
                
                # Plot background
                plot_bgcolor='white',
                paper_bgcolor='white',
                
                # Axes styling
                xaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='#E0E0E0',
                    showline=True,
                    linewidth=1,
                    linecolor='#CCCCCC',
                    mirror=False,
                    ticks='outside',
                    tickfont=dict(size=10, color="#666666")
                ),
                yaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='#E0E0E0',
                    showline=True,
                    linewidth=1,
                    linecolor='#CCCCCC',
                    mirror=False,
                    ticks='outside',
                    tickfont=dict(size=10, color="#666666")
                ),
                
                # Legend styling
                legend=dict(
                    bgcolor='rgba(255,255,255,0.9)',
                    bordercolor='#CCCCCC',
                    borderwidth=1,
                    font=dict(size=10)
                ),
                
                # Color scheme
                colorway=[
                    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
                ],
                
                # Margins
                margin=dict(l=60, r=30, t=60, b=60),
                
                # Hover styling
                hoverlabel=dict(
                    bgcolor="white",
                    font_size=11,
                    font_family="Arial, sans-serif"
                )
            )
        )
        
        # Register the custom template
        pio.templates["custom_jupyter"] = custom_template
        pio.templates.default = "custom_jupyter"
        
        # Set default figure size
        pio.kaleido.scope.default_width = 800
        pio.kaleido.scope.default_height = 500
        
    except ImportError:
        print("⚠️ Plotly not available - skipping Plotly configuration")

def apply_custom_css():
    """Apply custom CSS styling to improve notebook appearance"""
    try:
        from IPython.display import HTML, display
        
        css_styling = """
        <style>
        /* Enhanced notebook styling for better chart display */
        
        /* Main content area */
        .jp-Notebook {
            background-color: #fafafa;
        }
        
        /* Code cells */
        .jp-CodeCell .jp-Cell-inputWrapper {
            background-color: #f8f9fa;
            border-left: 3px solid #007acc;
        }
        
        /* Output areas */
        .jp-OutputArea {
            background-color: white;
            border-radius: 4px;
            margin: 10px 0;
            padding: 10px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        /* Chart containers */
        .plotly-graph-div {
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 10px 0;
        }
        
        /* Matplotlib figures */
        .jp-OutputArea img {
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 10px 0;
        }
        
        /* Headers */
        .jp-MarkdownCell h1 {
            color: #1a73e8;
            border-bottom: 2px solid #1a73e8;
            padding-bottom: 10px;
        }
        
        .jp-MarkdownCell h2 {
            color: #1967d2;
            border-bottom: 1px solid #e8eaed;
            padding-bottom: 5px;
        }
        
        .jp-MarkdownCell h3 {
            color: #5f6368;
        }
        
        /* Code syntax highlighting improvements */
        .cm-s-jupyter .cm-keyword { color: #d73a49; }
        .cm-s-jupyter .cm-string { color: #032f62; }
        .cm-s-jupyter .cm-comment { color: #6a737d; }
        .cm-s-jupyter .cm-number { color: #005cc5; }
        
        /* Table styling */
        .dataframe {
            border-collapse: collapse;
            margin: 10px 0;
            font-size: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            border-radius: 4px;
            overflow: hidden;
        }
        
        .dataframe th {
            background-color: #f1f3f4;
            color: #3c4043;
            font-weight: 500;
            padding: 8px 12px;
            text-align: left;
            border-bottom: 1px solid #dadce0;
        }
        
        .dataframe td {
            padding: 8px 12px;
            border-bottom: 1px solid #f1f3f4;
        }
        
        .dataframe tr:hover {
            background-color: #f8f9fa;
        }
        
        /* Progress bars */
        .progress {
            background-color: #e8eaed;
            border-radius: 4px;
            overflow: hidden;
            height: 8px;
        }
        
        .progress-bar {
            background-color: #1a73e8;
            height: 100%;
            transition: width 0.3s ease;
        }
        
        /* Alert boxes */
        .alert {
            padding: 12px 16px;
            margin: 10px 0;
            border-radius: 4px;
            border-left: 4px solid;
        }
        
        .alert-info {
            background-color: #e8f0fe;
            border-left-color: #1a73e8;
            color: #1967d2;
        }
        
        .alert-success {
            background-color: #e6f4ea;
            border-left-color: #34a853;
            color: #137333;
        }
        
        .alert-warning {
            background-color: #fef7e0;
            border-left-color: #fbbc04;
            color: #ea8600;
        }
        
        .alert-error {
            background-color: #fce8e6;
            border-left-color: #ea4335;
            color: #d93025;
        }
        
        /* Performance metrics styling */
        .metric-card {
            background: white;
            border-radius: 8px;
            padding: 16px;
            margin: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            border-left: 4px solid #1a73e8;
        }
        
        .metric-title {
            font-size: 14px;
            color: #5f6368;
            margin-bottom: 4px;
        }
        
        .metric-value {
            font-size: 24px;
            font-weight: 500;
            color: #3c4043;
        }
        
        .metric-unit {
            font-size: 12px;
            color: #5f6368;
        }
        </style>
        """
        
        display(HTML(css_styling))
        
    except ImportError:
        print("⚠️ IPython not available - skipping CSS styling")

def create_styled_chart_wrapper():
    """Create helper functions for styled charts"""
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        import matplotlib.pyplot as plt
        
        def styled_plotly_chart(fig, title=None, width=800, height=500):
            """Apply consistent styling to Plotly charts"""
            if title:
                fig.update_layout(title=dict(text=title, x=0.5, xanchor='center'))
            
            fig.update_layout(
                width=width,
                height=height,
                showlegend=True,
                margin=dict(l=60, r=30, t=60, b=60)
            )
            
            return fig
        
        def styled_matplotlib_chart(figsize=(12, 8), title=None):
            """Create a styled matplotlib figure"""
            fig, ax = plt.subplots(figsize=figsize)
            
            if title:
                ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
            
            # Apply grid
            ax.grid(True, alpha=0.3)
            ax.set_axisbelow(True)
            
            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            return fig, ax
        
        # Make functions globally available
        globals()['styled_plotly_chart'] = styled_plotly_chart
        globals()['styled_matplotlib_chart'] = styled_matplotlib_chart
        
    except ImportError:
        pass

def display_performance_metrics(metrics_dict, title="Performance Metrics"):
    """Display performance metrics in styled cards"""
    try:
        from IPython.display import HTML, display
        
        cards_html = f"<h3>{title}</h3><div style='display: flex; flex-wrap: wrap;'>"
        
        for metric_name, metric_value in metrics_dict.items():
            # Format the value based on type
            if isinstance(metric_value, float):
                if metric_value < 1:
                    formatted_value = f"{metric_value:.3f}"
                    unit = "seconds"
                elif metric_value < 60:
                    formatted_value = f"{metric_value:.2f}"
                    unit = "seconds"
                else:
                    formatted_value = f"{metric_value/60:.1f}"
                    unit = "minutes"
            else:
                formatted_value = str(metric_value)
                unit = ""
            
            cards_html += f"""
            <div class="metric-card">
                <div class="metric-title">{metric_name}</div>
                <div class="metric-value">{formatted_value} <span class="metric-unit">{unit}</span></div>
            </div>
            """
        
        cards_html += "</div>"
        display(HTML(cards_html))
        
    except ImportError:
        print("Performance metrics:", metrics_dict)

# Initialize styling when module is imported
if __name__ != "__main__":
    try:
        setup_notebook_styling()
        create_styled_chart_wrapper()
    except Exception as e:
        print(f"⚠️ Could not fully initialize styling: {e}")