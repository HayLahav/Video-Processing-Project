import cv2
import numpy as np
from utils import *


def moving_average(curve, radius):
    window_size = 2 * radius + 1
    # Define the filter
    f = np.ones(window_size) / window_size
    # Add padding to the boundaries
    curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
    # Apply convolution
    curve_smoothed = np.convolve(curve_pad, f, mode='same')
    # Remove padding
    curve_smoothed = curve_smoothed[radius:-radius]
    # Return smoothed curve
    return curve_smoothed


def fix_border(frame):
    s = frame.shape
    # Scale the image 4% without moving the center
    T = cv2.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, 1.04)
    frame = cv2.warpAffine(frame, T, (s[1], s[0]))
    return frame


def smooth(trajectory, radius):
    smoothed_trajectory = np.copy(trajectory)
    # Filter the x, y and angle curves
    for i in range(3):
        smoothed_trajectory[:, i] = moving_average(trajectory[:, i], radius=radius)
    return smoothed_trajectory


def stabilize_video(input_path, output_path, smoothing_radius):
    # Read input video
    cap = cv2.VideoCapture(input_path)
    params = get_video_parameters(cap)

    # Get frame count
    n_frames = params["frame_count"]

    # Get width and height of video stream
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec for output video
    #fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Set up output video
    #out = cv2.VideoWriter(output_path, fourcc, 30, (w, h))

    # Read first frame
    success, prev = cap.read()

    if not success:
        # Handle the case when the first frame cannot be read
        print("Error: Failed to read the first frame.")
        return

    # Check if the frame is empty or invalid
    if prev is None or prev.size == 0:
        print("Error: Empty or invalid frame.")
        return

    # Convert frame to grayscale
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    # Pre-define transformation-store array
    transforms = np.zeros((n_frames - 1, 3), np.float32)

    for i in range(n_frames - 2):
        # Detect feature points in previous frame
        prev_pts = cv2.goodFeaturesToTrack(prev_gray,
                                           maxCorners=200,
                                           qualityLevel=0.01,
                                           minDistance=30,
                                           blockSize=3)

        # Read next frame
        success, curr = cap.read()
        if not success:
            break

        # Convert to grayscale
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

        # Calculate new feature points based on optical flow
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

        # Filter only valid points
        idx = np.where(status == 1)[0]
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]

        # Find transformation matrix
        m = cv2.estimateAffine2D(prev_pts, curr_pts)[0]

        # Extract translation and rotation
        dx = m[0, 2]
        dy = m[1, 2]
        da = np.arctan2(m[1, 0], m[0, 0])

        # Store transformation
        transforms[i] = [dx, dy, da]

        # Move to next frame
        prev_gray = curr_gray

        print("Frame: {}/{} - Tracked points: {}".format(i, n_frames, len(prev_pts)))

    # Compute trajectory using cumulative sum of transformations
    trajectory = np.cumsum(transforms, axis=0)

    # Smooth trajectory
    smoothed_trajectory = smooth(trajectory, smoothing_radius)

    # Calculate difference between smoothed and original trajectory
    difference = smoothed_trajectory - trajectory

    # Calculate newer transformation array
    transforms_smooth = transforms + difference

    # Reset stream to first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frames_list = []
    # Write n_frames-1 transformed frames
    for i in range(n_frames - 2):
        # Read next frame
        success, frame = cap.read()
        if not success:
            break

        # Extract transformations from the new transformation array
        dx = transforms_smooth[i, 0]
        dy = transforms_smooth[i, 1]
        da = transforms_smooth[i, 2]

        # Reconstruct transformation matrix accordingly to new values
        m = np.zeros((2, 3), np.float32)
        m[0, 0] = np.cos(da)
        m[0, 1] = -np.sin(da)
        m[1, 0] = np.sin(da)
        m[1, 1] = np.cos(da)
        m[0, 2] = dx
        m[1, 2] = dy

        # Apply affine wrapping to the given frame
        frame_stabilized = cv2.warpAffine(frame, m, (w, h))

        # Fix border artifacts
        #frame_stabilized = fix_border(frame_stabilized)

        # Write the frame to the file
        frame_out = cv2.hconcat([frame, frame_stabilized])

        # If the image is too big, resize it.
        # if frame_out.shape[1] > 1920:
        #     frame_out = cv2.resize(frame_out, (frame_out.shape[1] // 2, frame_out.shape[0] // 2))

        frames_list.append(frame_stabilized)
        # Display the result
        # cv2.imshow("Before and After", frame_out)
        # cv2.waitKey(10)
        #out.write(frame_out)
    create_video_from_frames(frames_list, output_path, params["fps"])

    # Release resources
    cap.release()
    cv2.destroyAllWindows()





    print("Video stabilization complete!")

if __name__ == "__main__":

    # Load video file
    input_video_name = 'inputs/INPUT.mp4'
    output_video_path = 'outputs/stabilize_hai.avi'
# Example usage
    stabilize_video(input_video_name, output_video_path, 30)
