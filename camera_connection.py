import pyrealsense2 as rs
print("pyrealsense2 installed successfully!")

# Try to get a list of connected devices
ctx = rs.context()
devices = ctx.query_devices()
print(f"Found {devices.size()} connected RealSense devices.")

# Print device information if any are connected
for i in range(devices.size()):
    device = devices[i]
    print(f"Device {i}: {device.get_info(rs.camera_info.name)}")
    print(f"Serial number: {device.get_info(rs.camera_info.serial_number)}")
