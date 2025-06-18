#!/bin/bash

echo "🚀 Setting up Data Processing Comparison environment..."

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Set up Jupyter configuration
echo "🔧 Configuring Jupyter..."
mkdir -p ~/.jupyter
cat > ~/.jupyter/jupyter_lab_config.py << EOF
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.port = 8888
c.ServerApp.open_browser = False
c.ServerApp.allow_root = True
c.ServerApp.token = ''
c.ServerApp.password = ''
EOF

# Create a quick start script
echo "📝 Creating quick start script..."
cat > ~/start-jupyter.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting Jupyter Lab..."
echo "📍 Access at: http://localhost:8888"
echo "📚 Open any .ipynb file to get started!"
echo ""
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
EOF
chmod +x ~/start-jupyter.sh

# Set up environment variables for Spark
echo "⚡ Configuring Spark environment..."
echo 'export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64' >> ~/.bashrc
echo 'export SPARK_HOME=/usr/local/lib/python3.12/site-packages/pyspark' >> ~/.bashrc

echo "✅ Setup complete!"
echo ""
echo "🎯 Quick Start:"
echo "  1. Run: ~/start-jupyter.sh"
echo "  2. Open any scenario notebook (scenario1.ipynb, scenario2.ipynb, etc.)"
echo "  3. Run the cells to see data processing comparisons!"
echo ""
echo "📖 Available scenarios:"
echo "  - scenario1.ipynb: Jupyter Data Scientist workflow"
echo "  - scenario2.ipynb: Production ETL Pipeline"
echo "  - scenario3.ipynb: Real-time Analytics Dashboard"
echo "  - scenario4.ipynb: ML Feature Pipeline"
echo "  - overview.ipynb: Complete comparison overview"