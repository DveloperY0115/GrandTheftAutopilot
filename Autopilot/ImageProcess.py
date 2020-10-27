import numpy as np
import cv2


def set_ROI(frame, vertices):
    """
    Set the region of interest in the given frame designated as set of vertices.
    :param frame: a frame to set ROI on
    :param vertices: Vertex points of ROI
    :return: Masked frame
    """
    # Create a mask of the size of input img
    mask = np.zeros_like(frame)

    # Fill the mask with designated vertices
    cv2.fillPoly(mask, vertices, 255)

    # Mask the img
    masked = cv2.bitwise_and(frame, mask)
    return masked

"""
def draw_lanes(frame, lines, color=[0, 255, 255], thickness=3):

    try:
        # create an empty list for y coordinates of detected lines
        y_coords = []

        for line in lines:
            for line_component in line:
                y_coords += [line_component[1], line_component[3]]
        min_y = min(y_coords)
        max_y = 600
        new_lines = []


    for line in lines:
        coords = line[0]
        cv2.line(frame, (coords[0], coords[1]), (coords[2], coords[3]),
                 [255, 255, 255], 3)
"""

def process_frame(frame):
    """
    Processes the given frame using a set of elementary processing algorithms
    - RGB to Grayscale
    - Edge detection
    - Gaussian blur
    :param frame: a frame to do processing on
    :return: processed frame
    """
    original_frame = frame

    # convert to gray
    processed_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # edge detection
    processed_frame = cv2.Canny(processed_frame, threshold1=200, threshold2=300)

    # apply Gaussian blur for clear line detection
    processed_frame = cv2.GaussianBlur(processed_frame, (5, 5), 0)

    mask_vertices = np.array([[10, 550], [10, 300], [300, 200],
                              [500, 200], [800, 300], [800, 550]])
    processed_frame = set_ROI(processed_frame, [mask_vertices])

    # code for line detection - Provided by opencv library
    lines = cv2.HoughLinesP(processed_frame, 1, np.pi/180, 180, 20, 15)
    draw_lanes(processed_frame, lines)
    return processed_frame


def process_default(frame):
    """
    Default method for frame processing
    Do nothing
    :param frame: a frame to do processing on
    :return: same as input frame
    """
    original_frame = frame
    return original_frame
