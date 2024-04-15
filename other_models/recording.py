import pyrealsense2 as rs
import numpy as np
import cv2

def main():
    # Create a pipeline
    pipeline = rs.pipeline()

    # Create a config and configure the pipeline to stream RGB
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)  # Adjust resolution and frame rate as needed

    # Start streaming
    pipeline.start(config)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('./data/output.avi', fourcc, 30.0, (640, 480))  # Adjust output filename and parameters as needed

    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()

            # Get RGB frame
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())

            # Write the frame to the output video file
            out.write(color_image)

            # Display the resulting frame
            '''
            cv2.imshow('RealSense RGB', color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            '''

    finally:
        # Stop streaming
        pipeline.stop()
        
        # Release the VideoWriter
        out.release()

        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

