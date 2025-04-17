# FedVLA


https://github.com/user-attachments/assets/3d519e52-d234-48bb-b0f8-03087c37d43b

sudo apt-key adv --keyserver keys.gnupg.net --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE || sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE
sudo add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo focal main" -u
sudo apt-get update
sudo apt-get install librealsense2-dkms librealsense2-utils librealsense2-dev


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
