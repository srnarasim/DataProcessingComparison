"""
Notebook Setup Cell Template
============================

Copy and paste this code into the first cell of any notebook in the 
DataProcessingComparison project to ensure optimal chart styling and display.

This cell should be run before any plotting or visualization code.
"""

# =============================================================================
# NOTEBOOK STYLING SETUP - Run this cell first!
# =============================================================================

# Import styling module
try:
    from jupyter_styling import setup_notebook_styling, create_performance_badge
    
    # Setup professional styling (change 'professional' to 'dark', 'minimal', or 'colorful' as needed)
    styling_config = setup_notebook_styling(
        style='professional',  # Options: 'professional', 'dark', 'minimal', 'colorful'
        dpi=150,              # Higher DPI for crisp charts (100 for standard, 150+ for high-DPI)
        enable_plotly=True    # Enable plotly configuration
    )
    
    print("🎨 Notebook styling configured successfully!")
    print(f"📊 Style: {styling_config['style']}")
    print(f"🖥️  DPI: {styling_config['dpi']}")
    print(f"📈 Matplotlib backend: {styling_config['matplotlib_backend']}")
    
except ImportError:
    print("⚠️ jupyter_styling module not found. Using basic configuration...")
    
    # Fallback basic styling
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    
    # Suppress warnings
    warnings.filterwarnings('ignore')
    
    # Basic matplotlib configuration
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['savefig.dpi'] = 150
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    
    # Basic seaborn styling
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    sns.set_context("notebook", font_scale=1.1)
    
    # Enable inline plotting
    from IPython import get_ipython
    ipython = get_ipython()
    if ipython:
        ipython.magic('matplotlib inline')
    
    print("✅ Basic styling applied")

# Import common libraries with styling
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configure pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)

# Try to import and configure plotly
try:
    import plotly.graph_objects as go
    import plotly.express as px
    import plotly.io as pio
    
    # Configure plotly for Jupyter
    pio.renderers.default = "notebook"
    
    # Plotly configuration for better charts
    plotly_config = {
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['pan2d', 'lasso2d'],
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'chart',
            'height': 600,
            'width': 1000,
            'scale': 2
        }
    }
    
    print("📊 Plotly configured for interactive charts")
    
except ImportError:
    print("⚠️ Plotly not available - static charts only")
    plotly_config = None

# Display styling information
from IPython.display import HTML, display

display(HTML("""
<div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
            color: white; padding: 15px; border-radius: 10px; margin: 10px 0;">
    <h3 style="margin: 0; color: white;">📊 DataProcessingComparison - Notebook Ready!</h3>
    <p style="margin: 5px 0 0 0; opacity: 0.9;">
        Charts and visualizations are now optimized for your local Jupyter environment.
        All styling has been configured for professional-quality output.
    </p>
</div>
"""))

print("\n🚀 Notebook setup complete! You can now run your analysis cells.")
print("💡 Tip: Charts will automatically use the configured styling for consistent, professional appearance.")