#!/usr/bin/env python3
"""
Styling Demo - DataProcessingComparison
=======================================

This script demonstrates the difference between default Jupyter styling
and the enhanced professional styling system.

Run this script to see before/after examples of chart styling.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from jupyter_styling import setup_notebook_styling

def create_sample_data():
    """Create sample data for demonstration."""
    np.random.seed(42)
    
    # Performance comparison data
    tools = ['Pandas', 'Polars', 'DuckDB', 'Spark', 'TabsData']
    execution_times = [2.1, 0.8, 0.6, 1.5, 0.9]
    memory_usage = [450, 120, 80, 300, 150]
    
    # Time series data
    dates = pd.date_range('2024-01-01', periods=30, freq='D')
    performance_data = pd.DataFrame({
        'Date': dates,
        'Pandas': np.random.normal(2.1, 0.3, 30),
        'Polars': np.random.normal(0.8, 0.1, 30),
        'DuckDB': np.random.normal(0.6, 0.1, 30),
        'Spark': np.random.normal(1.5, 0.2, 30),
        'TabsData': np.random.normal(0.9, 0.15, 30)
    })
    
    return tools, execution_times, memory_usage, performance_data

def demo_default_styling():
    """Show charts with default matplotlib/seaborn styling."""
    print("📊 Creating charts with DEFAULT styling...")
    
    # Reset to default styling
    plt.rcdefaults()
    sns.reset_defaults()
    
    tools, exec_times, memory, perf_data = create_sample_data()
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Default Styling - Before Enhancement', fontsize=16)
    
    # Bar chart
    ax1.bar(tools, exec_times)
    ax1.set_title('Execution Time Comparison')
    ax1.set_ylabel('Time (seconds)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Scatter plot
    ax2.scatter(memory, exec_times, s=100)
    ax2.set_xlabel('Memory Usage (MB)')
    ax2.set_ylabel('Execution Time (seconds)')
    ax2.set_title('Memory vs Performance')
    
    # Line plot
    for tool in tools:
        ax3.plot(perf_data['Date'], perf_data[tool], label=tool, linewidth=1)
    ax3.set_title('Performance Over Time')
    ax3.set_ylabel('Time (seconds)')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.tick_params(axis='x', rotation=45)
    
    # Heatmap data
    correlation_data = perf_data[tools].corr()
    im = ax4.imshow(correlation_data, cmap='viridis', aspect='auto')
    ax4.set_xticks(range(len(tools)))
    ax4.set_yticks(range(len(tools)))
    ax4.set_xticklabels(tools, rotation=45)
    ax4.set_yticklabels(tools)
    ax4.set_title('Tool Correlation Matrix')
    
    plt.tight_layout()
    plt.savefig('demo_default_styling.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✅ Default styling demo saved as 'demo_default_styling.png'")

def demo_professional_styling():
    """Show charts with professional styling system."""
    print("\n🎨 Creating charts with PROFESSIONAL styling...")
    
    # Apply professional styling
    setup_notebook_styling(style='professional', dpi=150)
    
    tools, exec_times, memory, perf_data = create_sample_data()
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Professional Styling - After Enhancement', fontsize=16, fontweight='bold')
    
    # Enhanced bar chart
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    bars = ax1.bar(tools, exec_times, color=colors)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}s', ha='center', va='bottom', fontweight='bold')
    
    ax1.set_title('Execution Time Comparison', fontweight='bold')
    ax1.set_ylabel('Time (seconds)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Enhanced scatter plot
    scatter = ax2.scatter(memory, exec_times, s=150, c=colors, alpha=0.7, edgecolors='black', linewidth=1)
    ax2.set_xlabel('Memory Usage (MB)')
    ax2.set_ylabel('Execution Time (seconds)')
    ax2.set_title('Memory vs Performance', fontweight='bold')
    
    # Add tool labels to points
    for i, tool in enumerate(tools):
        ax2.annotate(tool, (memory[i], exec_times[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    # Enhanced line plot
    for i, tool in enumerate(tools):
        ax3.plot(perf_data['Date'], perf_data[tool], label=tool, 
                linewidth=2.5, color=colors[i], alpha=0.8)
    ax3.set_title('Performance Over Time', fontweight='bold')
    ax3.set_ylabel('Time (seconds)')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax3.tick_params(axis='x', rotation=45)
    
    # Enhanced heatmap
    correlation_data = perf_data[tools].corr()
    im = ax4.imshow(correlation_data, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)
    ax4.set_xticks(range(len(tools)))
    ax4.set_yticks(range(len(tools)))
    ax4.set_xticklabels(tools, rotation=45)
    ax4.set_yticklabels(tools)
    ax4.set_title('Tool Correlation Matrix', fontweight='bold')
    
    # Add correlation values to heatmap
    for i in range(len(tools)):
        for j in range(len(tools)):
            text = ax4.text(j, i, f'{correlation_data.iloc[i, j]:.2f}',
                           ha="center", va="center", color="black", fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax4, shrink=0.8)
    cbar.set_label('Correlation', rotation=270, labelpad=15)
    
    plt.tight_layout()
    plt.savefig('demo_professional_styling.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✅ Professional styling demo saved as 'demo_professional_styling.png'")

def demo_comparison():
    """Show side-by-side comparison of styling approaches."""
    print("\n🔍 Creating side-by-side comparison...")
    
    tools, exec_times, memory, perf_data = create_sample_data()
    
    # Create comparison figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Styling Comparison: Default vs Professional', fontsize=16, fontweight='bold')
    
    # Default styling (left)
    plt.rcdefaults()
    sns.reset_defaults()
    
    ax1.bar(tools, exec_times)
    ax1.set_title('Default Styling')
    ax1.set_ylabel('Execution Time (seconds)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Professional styling (right)
    setup_notebook_styling(style='professional', dpi=150)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    bars = ax2.bar(tools, exec_times, color=colors)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}s', ha='center', va='bottom', fontweight='bold')
    
    ax2.set_title('Professional Styling', fontweight='bold')
    ax2.set_ylabel('Execution Time (seconds)')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('demo_styling_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✅ Styling comparison saved as 'demo_styling_comparison.png'")

def main():
    """Run the complete styling demonstration."""
    print("🎨 DataProcessingComparison - Styling System Demo")
    print("=" * 50)
    
    # Demo 1: Default styling
    demo_default_styling()
    
    # Demo 2: Professional styling
    demo_professional_styling()
    
    # Demo 3: Side-by-side comparison
    demo_comparison()
    
    print("\n" + "=" * 50)
    print("🎉 Styling demo complete!")
    print("\n📊 Generated files:")
    print("   - demo_default_styling.png")
    print("   - demo_professional_styling.png") 
    print("   - demo_styling_comparison.png")
    print("\n💡 To use in your notebooks:")
    print("   from jupyter_styling import setup_notebook_styling")
    print("   setup_notebook_styling(style='professional')")

if __name__ == "__main__":
    main()