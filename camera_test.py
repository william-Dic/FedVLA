import pyrealsense2 as rs
import numpy as np
import cv2
import time

# Create a context object to manage devices
ctx = rs.context()
devices = ctx.query_devices()

# Check if any devices are connected
if devices.size() == 0:
    print("No RealSense devices detected! Please check your connection.")
    exit(1)
else:
    print(f"Found {devices.size()} RealSense device(s)")
    for i in range(devices.size()):
        device = devices[i]
        print(f"    Device {i}: {device.get_info(rs.camera_info.name)}")
        print(f"    Serial number: {device.get_info(rs.camera_info.serial_number)}")

# Create a pipeline and config
pipeline = rs.pipeline()
config = rs.config()

# Try to find a device that supports color streaming
found_rgb = False
for i in range(devices.size()):
    device = devices[i]
    # Get device product line (D400, SR300, etc.)
    product_line = device.get_info(rs.camera_info.product_line)
    serial = device.get_info(rs.camera_info.serial_number)
    
    # Enable color stream for this specific device
    config.enable_device(serial)
    try:
        # Enable RGB streaming - try different formats if one doesn't work
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        found_rgb = True
        print(f"Enabled color stream on device {serial}")
        break
    except Exception as e:
        print(f"Could not enable standard color stream: {e}")
        try:
            # Try different resolution
            config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
            found_rgb = True
            print(f"Enabled 720p color stream on device {serial}")
            break
        except Exception as e2:
            print(f"Could not enable 720p color stream: {e2}")
            continue

if not found_rgb:
    print("Could not find a device that supports color streaming!")
    exit(1)

# Start streaming with a timeout
print("Attempting to start pipeline...")
try:
    pipeline_profile = pipeline.start(config)
    print("Pipeline started successfully")
    
    # Get the selected device
    selected_device = pipeline_profile.get_device()
    print(f"Using device: {selected_device.get_info(rs.camera_info.name)}")
    
    # Print active streams
    active_streams = []
    for stream in pipeline_profile.get_streams():
        stream_type = stream.stream_type()
        if stream_type == rs.stream.color:
            active_streams.append("Color")
        elif stream_type == rs.stream.depth:
            active_streams.append("Depth")
        elif stream_type == rs.stream.infrared:
            active_streams.append("Infrared")
    print(f"Active streams: {', '.join(active_streams)}")
except Exception as e:
    print(f"Failed to start pipeline: {e}")
    exit(1)

# Main loop
try:
    frame_count = 0
    start_time = time.time()
    
    while True:
        # Wait for the next set of frames with timeout
        try:
            frames = pipeline.wait_for_frames(5000)  # 5 second timeout
            color_frame = frames.get_color_frame()
            
            if not color_frame:
                print("Received empty color frame")
                continue
                
            # Convert color frame to a numpy array and display it
            color_image = np.asanyarray(color_frame.get_data())
            
            # Check if array is valid
            if color_image.size == 0:
                print("Empty color image data")
                continue
                
            # Calculate FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= 5.0:  # Update FPS every 5 seconds
                fps = frame_count / elapsed_time
                print(f"FPS: {fps:.2f}")
                frame_count = 0
                start_time = time.time()
                
            # Display the image
            cv2.imshow("RealSense Camera", color_image)
            
            # Exit the loop when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
                
        except Exception as e:
            print(f"Error during frame capture: {e}")
            time.sleep(1)  # Wait a bit before trying again
            
finally:
    # Stop streaming and release resources
    print("Stopping pipeline...")
    pipeline.stop()
    cv2.destroyAllWindows()
    print("Resources released.")
