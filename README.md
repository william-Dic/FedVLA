# FedVLA


https://github.com/user-attachments/assets/3d519e52-d234-48bb-b0f8-03087c37d43b
# Intel RealSense D304 Camera - Python Setup Guide

This repository contains Python code examples and setup instructions for using the Intel RealSense D304 depth camera on Ubuntu 20.04.

## Prerequisites

- Ubuntu 20.04 LTS
- Intel RealSense D304 camera
- Python 3.8+ (comes with Ubuntu 20.04)
- USB 3.0 port (recommended for best performance)

## Installation

### 1. Install the Intel RealSense SDK 2.0

The RealSense SDK provides the necessary drivers and libraries to interface with the camera.

```bash
# Add the RealSense repository to your system
sudo apt-key adv --keyserver keys.gnupg.net --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE || sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE
sudo add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo focal main" -u

# Update the package list
sudo apt-get update

# Install the SDK packages
sudo apt-get install -y librealsense2-dkms librealsense2-utils librealsense2-dev
```

### 2. Install the Python packages

```bash
# Install the Python wrapper and dependencies
pip install pyrealsense2 numpy opencv-python
```
