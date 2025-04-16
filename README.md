# FedVLA


https://github.com/user-attachments/assets/3d519e52-d234-48bb-b0f8-03087c37d43b


# Intel RealSense D405 Camera: Python Setup Guide
Installation Steps
1. Install Intel RealSense SDK 2.0
Download and install the latest Intel RealSense SDK 2.0 from the official release page:
https://github.com/IntelRealSense/librealsense/releases
2. Set up a Compatible Python Environment
Option A: Using an officially supported Python version (recommended)
bash# Install Python 3.11 (or other supported version) from python.org or Microsoft Store
# Create a virtual environment
python3.11 -m venv .venv

# Activate the environment
# On Windows:
.venv\Scripts\activate

# Install required packages
pip install pyrealsense2
pip install numpy
pip install opencv-python
