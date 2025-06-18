#!/bin/bash

echo "🚀 Setting up Data Processing Comparison environment..."

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Generate sample data if it doesn't exist
echo "📊 Checking for sample data..."
if [ ! -d "data/transactions" ]; then
    echo "📊 Generating sample data for scenarios..."
    python generate_sample_data.py
else
    echo "✅ Sample data already exists"
fi

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

# Configure TabsData server
echo "🗄️ Configuring TabsData server..."
mkdir -p ~/.tabsdata

# Create TabsData server configuration
cat > ~/.tabsdata/config.yaml << EOF
server:
  host: "0.0.0.0"
  port: 8080
  log_level: "INFO"
storage:
  type: "local"
  path: "/tmp/tabsdata"
EOF

# Create TabsData server startup script
cat > ~/start-tabsdata.sh << 'EOF'
#!/bin/bash
echo "🗄️ Starting TabsData server..."
echo "📍 Server will be available at: http://localhost:8080"
echo "📊 TabsData UI and API endpoints ready"
echo ""

# Create storage directory
mkdir -p /tmp/tabsdata

# Start TabsData server using td CLI
if command -v td &> /dev/null; then
    td server start --port 8080 --host 0.0.0.0 > /tmp/tabsdata.log 2>&1 &
    TDSERVER_PID=$!
    echo "✅ TabsData server started (PID: $TDSERVER_PID)"
else
    echo "❌ TabsData CLI not found. Install with: pip install tabsdata"
    echo "Then run: td server start --port 8080"
fi

echo "📋 Server logs: /tmp/tabsdata.log"
echo "🔍 Check status: ps aux | grep td"
echo ""
EOF
chmod +x ~/start-tabsdata.sh

# Create combined startup script
cat > ~/start-all.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting all services for Data Processing Comparison..."
echo ""

# Start TabsData server first
echo "1️⃣ Starting TabsData server..."
~/start-tabsdata.sh

# Wait a moment for server to initialize
sleep 3

# Start Jupyter Lab
echo "2️⃣ Starting Jupyter Lab..."
echo "📍 Jupyter Lab: http://localhost:8888"
echo "📍 TabsData Server: http://localhost:8080"
echo ""
echo "📚 Open any .ipynb file to get started!"
echo "🗄️ TabsData server is ready for pub/sub operations"
echo ""

jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
EOF
chmod +x ~/start-all.sh

echo "✅ Setup complete!"
echo ""
echo "🎯 Quick Start Options:"
echo "  🚀 All services: ~/start-all.sh (Recommended)"
echo "  📊 Jupyter only: ~/start-jupyter.sh"
echo "  🗄️ TabsData only: ~/start-tabsdata.sh"
echo ""
echo "📖 Available scenarios:"
echo "  - overview.ipynb: Complete comparison overview"
echo "  - scenario1.ipynb: Jupyter Data Scientist workflow"
echo "  - scenario2.ipynb: Production ETL Pipeline (includes TabsData)"
echo "  - scenario3.ipynb: Real-time Analytics Dashboard"
echo "  - scenario4.ipynb: ML Feature Pipeline"
echo ""
echo "🌐 Service URLs:"
echo "  - Jupyter Lab: http://localhost:8888"
echo "  - TabsData Server: http://localhost:8080"