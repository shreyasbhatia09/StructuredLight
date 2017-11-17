# ================================================
# Skeleton codes for HW5
# Read the skeleton codes carefully and put all your
# codes into function "reconstruct_from_binary_patterns"
# ================================================

import cv2
import numpy as np
from math import log, ceil, floor
import matplotlib.pyplot as plt
import pickle
import sys


def help_message():
    # Note: it is assumed that "binary_codes_ids_codebook.pckl", "stereo_calibration.pckl",
    # and images folder are in the same root folder as your "generate_data.py" source file.
    # Same folder structure will be used when we test your program

    print("Usage: [Output_Directory]")
    print("[Output_Directory]")
    print("Where to put your output.xyz")
    print("Example usages:")
    print(sys.argv[0] + " ./")


def reconstruct_from_binary_patterns():
    scale_factor = 1.0
    ref_white = cv2.resize(cv2.imread("images/pattern000.jpg", cv2.IMREAD_GRAYSCALE) / 255.0, (0, 0), fx=scale_factor,
                           fy=scale_factor)
    ref_black = cv2.resize(cv2.imread("images/pattern001.jpg", cv2.IMREAD_GRAYSCALE) / 255.0, (0, 0), fx=scale_factor,
                           fy=scale_factor)
    ref_avg = (ref_white + ref_black) / 2.0
    ref_on = ref_avg + 0.05  # a threshold for ON pixels
    ref_off = ref_avg - 0.05  # add a small buffer region
    print ref_avg
    h, w = ref_white.shape

    # mask of pixels where there is projection
    proj_mask = (ref_white > (ref_black + 0.05))

    scan_bits = np.zeros((h, w), dtype=np.uint16)

    # analyze the binary patterns from the camera
    for i in range(0, 15):
        # read the file
        # patt = cv2.resize(cv2.imread("images/pattern%03d.jpg" % (i + 2)), (0, 0), fx=scale_factor, fy=scale_factor)
        patt_gray = cv2.resize(cv2.imread("images/pattern%03d.jpg" % (i + 2), cv2.IMREAD_GRAYSCALE) / 255.0, (0, 0),
                               fx=scale_factor, fy=scale_factor)
        # mask where the pixels are ON
        on_mask = (patt_gray > ref_on) & proj_mask

        # this code corresponds with the binary pattern code
        bit_code = np.uint16(1 << i)

        # TODO: populate scan_bits by putting the bit_code according to on_mask


    print("load codebook")
    # the codebook translates from <binary code> to (x,y) in projector screen space
    with open("binary_codes_ids_codebook.pckl", "r") as f:
        binary_codes_ids_codebook = pickle.load(f)

    camera_points = []
    projector_points = []
    for x in range(w):
        for y in range(h):
            if not proj_mask[y, x]:
                continue  # no projection here
            if scan_bits[y, x] not in binary_codes_ids_codebook:
                continue  # bad binary code

                # TODO: use binary_codes_ids_codebook[...] and scan_bits[y,x] to
                # TODO: find for the camera (x,y) the projector (p_x, p_y).
                # TODO: store your points in camera_points and projector_points

                # IMPORTANT!!! : due to differences in calibration and acquisition - divide the camera points by 2

    # now that we have 2D-2D correspondances, we can triangulate 3D points!

    # load the prepared stereo calibration between projector and camera
    with open("stereo_calibration.pckl", "r") as f:
        d = pickle.load(f)
        camera_K = d['camera_K']
        camera_d = d['camera_d']
        projector_K = d['projector_K']
        projector_d = d['projector_d']
        projector_R = d['projector_R']
        projector_t = d['projector_t']

        # TODO: use cv2.undistortPoints to get normalized points for camera, use camera_K and camera_d
        # TODO: use cv2.undistortPoints to get normalized points for projector, use projector_K and projector_d

        # TODO: use cv2.triangulatePoints to triangulate the normalized points
        # TODO: use cv2.convertPointsFromHomogeneous to get real 3D points
        # TODO: name the resulted 3D points as "points_3d"

        return points_3d


def write_3d_points(points_3d):
    # ===== DO NOT CHANGE THIS FUNCTION =====

    print("write output point cloud")
    print(points_3d.shape)
    output_name = sys.argv[1] + "output.xyz"
    with open(output_name, "w") as f:
        for p in points_3d:
            f.write("%d %d %d\n" % (p[0, 0], p[0, 1], p[0, 2]))

    return points_3d, camera_points, 2

if __name__ == '__main__':

    # ===== DO NOT CHANGE THIS FUNCTION =====

    # validate the input arguments
    if (len(sys.argv) != 2):
        help_message()
        sys.exit()

    points_3d = reconstruct_from_binary_patterns()
    write_3d_points(points_3d)
