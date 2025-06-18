# Development Container Configuration

This repository is configured with GitHub Codespaces and VS Code Dev Containers for a consistent development environment.

## Features

- **Python 3.12** - Latest stable Python version
- **Java 11** - Required for Apache Spark
- **Node.js 18** - For any web-based visualizations
- **Pre-installed Extensions**:
  - Python language support with Pylance
  - Jupyter notebook support
  - Code formatting (Black, isort)
  - Linting (Flake8, Ruff)
  - GitHub Copilot (if available)

## Getting Started

### Option 1: GitHub Codespaces (Recommended)
1. Go to the repository on GitHub
2. Click the green "Code" button
3. Select "Codespaces" tab
4. Click "Create codespace on main"

### Option 2: VS Code Dev Containers
1. Install VS Code and the "Dev Containers" extension
2. Clone this repository locally
3. Open in VS Code
4. When prompted, click "Reopen in Container"

## What's Included

The development environment automatically installs:
- All data processing libraries (Pandas, Polars, DuckDB, Spark)
- Jupyter Lab for interactive development
- Visualization libraries (Matplotlib, Seaborn, Plotly)
- Development tools (Black, Flake8, etc.)

## Port Forwarding

The following ports are automatically forwarded:
- **8888**: Jupyter Lab server
- **8000**: Development server
- **5000**: Flask/FastAPI applications

## Running the Notebooks

Once the container is ready:

```bash
# Start Jupyter Lab
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# Or run individual notebooks
jupyter notebook scenario1.ipynb
```

## Spark Configuration

Apache Spark is pre-configured with Java 11. The notebooks will automatically detect and configure Spark when needed.

## Troubleshooting

If you encounter issues:
1. Rebuild the container: Command Palette → "Dev Containers: Rebuild Container"
2. Check the container logs for any installation errors
3. Ensure you have sufficient resources allocated to the container