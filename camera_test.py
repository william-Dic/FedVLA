import pyrealsense2 as rs
import numpy as np
import cv2
import time

try:
    # 1. Check for available devices first
    ctx = rs.context()
    devices = ctx.query_devices()
    
    if devices.size() == 0:
        print("No RealSense devices detected!")
        exit(1)
    else:
        print(f"Found {devices.size()} RealSense device(s)")
        for i in range(devices.size()):
            device = devices[i]
            print(f"Device {i}: {device.get_info(rs.camera_info.name)}")
            print(f"Serial: {device.get_info(rs.camera_info.serial_number)}")
            print(f"Firmware: {device.get_info(rs.camera_info.firmware_version)}")
    
    # 2. Create a simpler configuration with lower resolution
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Try a lower resolution and framerate to reduce bandwidth requirements
    print("Configuring for 320x240 @ 15fps...")
    config.enable_stream(rs.stream.color, 320, 240, rs.format.bgr8, 15)
    
    # 3. Add advanced options to improve stability
    device = devices[0]  # Use the first device
    serial = device.get_info(rs.camera_info.serial_number)
    config.enable_device(serial)
    
    # 4. Start the pipeline with explicit error checking
    print(f"Starting pipeline for device {serial}...")
    try:
        profile = pipeline.start(config)
        print("Pipeline started successfully")
        
        # Get the actual device being used
        selected_device = profile.get_device()
        print(f"Using device: {selected_device.get_info(rs.camera_info.name)}")
        
        # 5. Set device advanced options
        sensors = selected_device.query_sensors()
        for sensor in sensors:
            print(f"Found sensor: {sensor.get_info(rs.camera_info.name)}")
            if sensor.supports(rs.option.enable_auto_exposure):
                print("Setting auto exposure")
                sensor.set_option(rs.option.enable_auto_exposure, 1)
    
    except Exception as e:
        print(f"Failed to start pipeline: {e}")
        exit(1)
    
    # 6. Main loop with more frequent frame checking
    print("Entering main loop...")
    frame_count = 0
    start_time = time.time()
    
    while True:
        try:
            # Use a shorter timeout to be more responsive
            frames = pipeline.wait_for_frames(1000)  # 1 second timeout
            color_frame = frames.get_color_frame()
            
            if not color_frame:
                print("No color frame received")
                continue
            
            # Process frame
            color_image = np.asanyarray(color_frame.get_data())
            
            # Display with minimal processing
            cv2.imshow("RealSense", color_image)
            
            # Count frames for FPS calculation
            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed >= 2.0:
                fps = frame_count / elapsed
                print(f"FPS: {fps:.1f}")
                frame_count = 0
                start_time = time.time()
            
            # Check for quit
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                print("User requested exit")
                break
                
        except Exception as e:
            print(f"Error in main loop: {e}")
            # Wait a bit before retrying
            time.sleep(0.1)
    
except Exception as e:
    print(f"Unhandled exception: {e}")

finally:
    # Clean up
    try:
        pipeline.stop()
        print("Pipeline stopped")
    except:
        pass
    
    cv2.destroyAllWindows()
    print("Resources released")
