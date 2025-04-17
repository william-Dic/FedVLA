import pyrealsense2 as rs
import numpy as np
import cv2

# Create a pipeline and configure the stream
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming from the camera
pipeline.start(config)

try:
    while True:
        # Wait for the next set of frames from the camera
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert color frame to a numpy array and display it
        color_image = np.asanyarray(color_frame.get_data())
        cv2.imshow("RealSense Camera", color_image)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    # Stop streaming and release resources
    pipeline.stop()
    cv2.destroyAllWindows()
