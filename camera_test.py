import pyrealsense2 as rs
import numpy as np
import cv2

# Print available devices
ctx = rs.context()
devices = ctx.query_devices()
print(f"Found {len(devices)} RealSense device(s)")

for i, dev in enumerate(devices):
    print(f"Device {i}: {dev.get_info(rs.camera_info.name)}")
    print(f"Serial: {dev.get_info(rs.camera_info.serial_number)}")
    print(f"Firmware: {dev.get_info(rs.camera_info.firmware_version)}")

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Try with the default supported configuration
# Instead of hardcoding resolution and fps, let's query what's supported
device = devices[0]
print(f"Configuring for device {device.get_info(rs.camera_info.serial_number)}...")

# Get the first depth sensor from the device
depth_sensor = device.first_depth_sensor()

# Get supported stream profiles
profiles = depth_sensor.get_stream_profiles()
depth_profiles = [p for p in profiles if p.stream_type() == rs.stream.depth]

# Sort profiles by resolution (highest first)
depth_profiles.sort(key=lambda p: p.as_video_stream_profile().width() * p.as_video_stream_profile().height(), reverse=True)

# Use the first (highest resolution) profile
if depth_profiles:
    depth_profile = depth_profiles[0].as_video_stream_profile()
    width = depth_profile.width()
    height = depth_profile.height()
    fps = depth_profile.fps()
    print(f"Using depth profile: {width}x{height} @ {fps}fps")
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    
    # Also try to enable color stream with same resolution if possible
    try:
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        print(f"Using color profile: {width}x{height} @ {fps}fps")
    except Exception as e:
        print(f"Couldn't enable color stream with same parameters: {e}")
        print("Trying default color stream parameters...")
        config.enable_stream(rs.stream.color)

print("Starting pipeline...")
try:
    pipeline.start(config)
    print("Pipeline started successfully!")
    
    try:
        while True:
            # Wait for a coherent pair of frames
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            
            # Try to get color frame, but don't fail if not available
            color_frame = None
            try:
                color_frame = frames.get_color_frame()
            except:
                pass
                
            if not depth_frame:
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            
            # Apply colormap on depth image
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), 
                cv2.COLORMAP_JET
            )
            
            # If we have color frame, show both
            if color_frame:
                color_image = np.asanyarray(color_frame.get_data())
                images = np.hstack((color_image, depth_colormap))
                cv2.imshow('RealSense D405', images)
            else:
                # Otherwise just show depth
                cv2.imshow('RealSense D405 (Depth Only)', depth_colormap)
                
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
                
    finally:
        print("Stopping pipeline...")
        pipeline.stop()
        print("Pipeline stopped.")
        
except Exception as e:
    print(f"Failed to start pipeline: {e}")
    if "Couldn't resolve requests" in str(e):
        print("\nTroubleshooting tips:")
        print("1. Try listing all supported profiles for your D405:")
        print("\nFor example, you could run:")
        print("for profile in depth_sensor.get_stream_profiles():")
        print("    if profile.stream_type() == rs.stream.depth:")
        print("        video_profile = profile.as_video_stream_profile()")
        print("        print(f'Depth: {video_profile.width()}x{video_profile.height()} @ {video_profile.fps()}fps')")

print("Resources released")
