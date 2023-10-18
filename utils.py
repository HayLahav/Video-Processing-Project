import cv2
import numpy as np

def get_video_parameters(capture: cv2.VideoCapture) -> dict:
    """Get an OpenCV capture object and extract its parameters.

    Args:
        capture: cv2.VideoCapture object.

    Returns:
        parameters: dict. Video parameters extracted from the video.

    """
    fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    return {"fourcc": fourcc, "fps": fps, "height": height, "width": width,
            "frame_count": frame_count}


def create_video_from_frames(output_frames, video_output_name, fps):
    OUTPUT_SIZE = (output_frames[0].shape[1], output_frames[0].shape[0])
    output_video = cv2.VideoWriter(video_output_name, cv2.VideoWriter_fourcc(*'XVID'),fps, OUTPUT_SIZE)
    for i in range(len(output_frames)):
        frame2write = output_frames[i]
        if len(frame2write.shape) < 3:
            frame2write = cv2.cvtColor(frame2write, cv2.COLOR_GRAY2RGB)
        frame2write = cv2.resize(np.uint8(frame2write), OUTPUT_SIZE)
        output_video.write(frame2write)
    output_video.release()
    cv2.destroyAllWindows()