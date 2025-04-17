import pyrealsense2 as rs
import numpy as np
import cv2

# Configure depth stream without specifying resolution/fps
pipeline = rs.pipeline()
config = rs.config()

# Enable streams without specifying any parameters - use device defaults
config.enable_stream(rs.stream.depth)  # Use default settings for depth

try:
    print("Starting pipeline with default settings...")
    pipeline.start(config)
    print("Pipeline started successfully!")
    
    try:
        while True:
            # Wait for frames
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            
            if not depth_frame:
                continue
                
            # Process depth data
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), 
                cv2.COLORMAP_JET
            )
            
            # Show image
            cv2.imshow('D405 Depth', depth_colormap)
            
            # Exit on ESC or 'q'
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                break
                
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Pipeline stopped cleanly")
        
except Exception as e:
    print(f"Error: {e}")

print("Program ended")
