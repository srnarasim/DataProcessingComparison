#!/usr/bin/env python3
"""
Test the overview notebook styling setup specifically.
"""

def test_overview_styling():
    """Test the styling setup that was causing issues."""
    print("🔍 Testing overview notebook styling setup...")
    
    try:
        # Test the exact code from the overview notebook
        from jupyter_styling import setup_notebook_styling
        
        # Setup professional styling
        styling_config = setup_notebook_styling(
            style='professional',  # Professional, clean styling
            dpi=150,              # High-DPI for crisp charts
            enable_plotly=True    # Enable interactive plotly charts
        )
        
        print("🎨 Professional notebook styling configured!")
        print(f"✅ Styling config: {styling_config}")
        return True
        
    except ImportError:
        print("⚠️ Using basic styling (jupyter_styling module not found)")
        
        # Fallback basic styling
        import matplotlib.pyplot as plt
        import seaborn as sns
        import warnings
        warnings.filterwarnings('ignore')
        
        # Enhanced matplotlib configuration
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['figure.dpi'] = 150
        plt.rcParams['savefig.dpi'] = 150
        plt.rcParams['font.size'] = 11
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        
        # Seaborn styling
        sns.set_style("whitegrid")
        sns.set_palette("husl")
        sns.set_context("notebook", font_scale=1.1)
        
        print("✅ Basic styling configured successfully")
        return True
        
    except Exception as e:
        print(f"❌ Styling setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_overview_styling()
    if success:
        print("\n🎉 Overview notebook styling test passed!")
        exit(0)
    else:
        print("\n❌ Overview notebook styling test failed!")
        exit(1)