import pyrealsense2 as rs
import numpy as np
import cv2
import time

# Create a context object to discover RealSense devices
ctx = rs.context()
devices = ctx.query_devices()

if len(devices) == 0:
    print("No RealSense devices detected!")
    exit()

print(f"Found {len(devices)} RealSense device(s)")
device = devices[0]

print(f"Device: {device.get_info(rs.camera_info.name)}")
print(f"Serial: {device.get_info(rs.camera_info.serial_number)}")
print(f"Firmware: {device.get_info(rs.camera_info.firmware_version)}")

# Query available sensors
sensors = device.query_sensors()
print(f"Device has {len(sensors)} sensors")

# Find and check depth sensor capabilities
depth_sensor = None
for sensor in sensors:
    if sensor.is_depth_sensor():
        depth_sensor = sensor
        print(f"Found depth sensor: {sensor.get_info(rs.camera_info.name)}")
        break

if not depth_sensor:
    print("No depth sensor found!")
    exit()

# Get supported stream profiles for depth
depth_profiles = []
for profile in depth_sensor.get_stream_profiles():
    if profile.stream_type() == rs.stream.depth:
        try:
            vp = profile.as_video_stream_profile()
            depth_profiles.append((vp.width(), vp.height(), vp.fps(), profile.format()))
        except:
            pass

# Sort profiles by resolution (width * height)
depth_profiles.sort(key=lambda p: p[0] * p[1])

if not depth_profiles:
    print("No depth profiles found!")
    exit()

print("\nSupported depth profiles:")
for i, (width, height, fps, fmt) in enumerate(depth_profiles):
    print(f"{i}: {width}x{height} @ {fps}fps ({fmt.name})")

# Try each profile starting from the lowest resolution
print("\nTrying each depth profile...")

for i, (width, height, fps, fmt) in enumerate(depth_profiles):
    print(f"\nTrying profile {i}: {width}x{height} @ {fps}fps ({fmt.name})")
    
    # Configure pipeline with current profile
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, width, height, fmt, fps)
    
    try:
        print("Starting pipeline...")
        pipeline.start(config)
        print("SUCCESS! Pipeline started with this profile.")
        
        # Try to get a few frames
        print("Trying to get frames...")
        frame_count = 0
        start_time = time.time()
        
        try:
            # Try to get 10 frames or run for 3 seconds max
            while frame_count < 10 and (time.time() - start_time) < 3:
                frames = pipeline.wait_for_frames(timeout_ms=1000)
                depth_frame = frames.get_depth_frame()
                
                if depth_frame:
                    frame_count += 1
                    print(f"Received frame {frame_count}")
            
            if frame_count > 0:
                print(f"Successfully received {frame_count} frames")
            else:
                print("No frames received within timeout")
                
        except Exception as e:
            print(f"Error while getting frames: {e}")
        
        # Stop the pipeline
        pipeline.stop()
        print("Pipeline stopped")
        
        # If we successfully got frames, we can exit the loop
        if frame_count > 0:
            print(f"\nFound working configuration: {width}x{height} @ {fps}fps ({fmt.name})")
            break
            
    except Exception as e:
        print(f"Failed to start pipeline: {e}")
        # Continue to the next profile

print("\nDiagnostics complete")
