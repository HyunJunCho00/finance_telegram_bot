#!/bin/bash

# Define paths and user
APP_DIR="/opt/app"
VENV_DIR="$APP_DIR/venv"
USER="qkdtoddl1"
GROUP="qkdtoddl1"

echo "Creating systemd service files..."

# 1. Collector Service
cat <<EOF > /tmp/bot-collector.service
[Unit]
Description=Crypto Bot - Continuous Data Collector
After=network.target

[Service]
User=$USER
Group=$GROUP
WorkingDirectory=$APP_DIR
Environment="PATH=$VENV_DIR/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
Environment="USE_SECRET_MANAGER=true"
ExecStart=$VENV_DIR/bin/python $APP_DIR/app_collector.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# 2. Trading/Analysis Service
cat <<EOF > /tmp/bot-trading.service
[Unit]
Description=Crypto Bot - Trading and Analysis
After=network.target

[Service]
User=$USER
Group=$GROUP
WorkingDirectory=$APP_DIR
Environment="PATH=$VENV_DIR/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
Environment="USE_SECRET_MANAGER=true"
ExecStart=$VENV_DIR/bin/python $APP_DIR/app_trading.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# 3. Bot/UI Service
cat <<EOF > /tmp/bot-ui.service
[Unit]
Description=Crypto Bot - Telegram UI
After=network.target

[Service]
User=$USER
Group=$GROUP
WorkingDirectory=$APP_DIR
Environment="PATH=$VENV_DIR/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
Environment="USE_SECRET_MANAGER=true"
ExecStart=$VENV_DIR/bin/python $APP_DIR/app_bot.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

echo "Fixing directory permissions for data and cache..."
# Ensure directories exist and belong to the correct user
sudo mkdir -p $APP_DIR/data $APP_DIR/cache
sudo chown -R $USER:$GROUP $APP_DIR/data $APP_DIR/cache
sudo chmod -R 775 $APP_DIR/data $APP_DIR/cache

echo "Moving service files to /etc/systemd/system/ (requires sudo)..."
sudo mv /tmp/bot-*.service /etc/systemd/system/

echo "Reloading systemd daemon..."
sudo systemctl daemon-reload

echo "Enabling services (start on boot)..."
sudo systemctl enable bot-collector.service
sudo systemctl enable bot-trading.service
sudo systemctl enable bot-ui.service

echo "Restarting all services to apply changes..."
sudo systemctl restart bot-collector bot-trading bot-ui

echo "Deployment complete!"
echo "Services are now running and directory permissions are fixed."
echo ""
echo "To view logs, run:"
echo "sudo journalctl -u bot-collector -f"
