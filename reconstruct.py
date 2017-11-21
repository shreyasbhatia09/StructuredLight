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


def display_image(img):
    cv2.imshow('display', img)
    cv2.waitKey(0)


def reconstruct_from_binary_patterns():
    scale_factor = 1.0
    ref_white = cv2.resize(cv2.imread("images/pattern000.jpg", cv2.IMREAD_GRAYSCALE) / 255.0, (0, 0), fx=scale_factor,
                           fy=scale_factor)
    ref_black = cv2.resize(cv2.imread("images/pattern001.jpg", cv2.IMREAD_GRAYSCALE) / 255.0, (0, 0), fx=scale_factor,
                           fy=scale_factor)
    cam_color = cv2.imread("images/pattern001.jpg")
    ref_avg = (ref_white + ref_black) / 2.0
    ref_on = ref_avg + 0.05  # a threshold for ON pixels
    ref_off = ref_avg - 0.05  # add a small buffer region

    h, w = ref_white.shape

    # mask of pixels where there is projection
    proj_mask = (ref_white > (ref_black + 0.05))

    scan_bits = np.zeros((h, w), dtype=np.uint16)

    # analyze the binary patterns from the camera
    for i in range(0, 15):
        # read the file
        patt_gray = cv2.resize(cv2.imread("images/pattern%03d.jpg" % (i + 2), cv2.IMREAD_GRAYSCALE) / 255.0, (0, 0),
                               fx=scale_factor, fy=scale_factor)

        # mask where the pixels are ON
        on_mask = (patt_gray > ref_on) & proj_mask

        # this code corresponds with the binary pattern code
        bit_code = np.uint16(1 << i)

        # TODO: populate scan_bits by putting the bit_code according to on_mask
        scan_bits[on_mask == True] += bit_code

    print("load codebook")
    # the codebook translates from <binary code> to (x,y) in projector screen space
    with open("binary_codes_ids_codebook.pckl", "r") as f:
        binary_codes_ids_codebook = pickle.load(f)

    camera_points = []
    projector_points = []
    corr_img = np.zeros((proj_mask.shape[0], proj_mask.shape[1], 3))
    cam_rgb = []
    for x in range(w):
        for y in range(h):
            if not proj_mask[y, x]:
                continue  # no projection here
            if scan_bits[y, x] not in binary_codes_ids_codebook:
                continue  # bad binary code

            # TODO: use binary_codes_ids_codebook[...] and scan_bits[y,x] to
            # TODO: find for the camera (x,y) the projector (p_x, p_y).
            # TODO: store your points in camera_points and projector_points
            proj_x, proj_y = binary_codes_ids_codebook[scan_bits[y, x]]

            # IMPORTANT!!! : due to differences in calibration and acquisition - divide the camera points by 2
            if proj_x >= 1279 or proj_y >= 799:  # filter
                 continue
            projector_points.append([[proj_x, proj_y]])
            camera_points.append([[x / 2.0, y / 2.0]])
            cam_rgb.append([ cam_color[y, x] ])
            corr_img[y, x, 2] = np.uint8((proj_x / 1279.0) * 255.0)
            corr_img[y, x, 1] = np.uint8((proj_y / 799.0) * 255.0)

    # cv2.imshow("corr_img", np.array(corr_img).astype('uint8'))
    correspondant_img_path = sys.argv[1] + "correspondence.jpg"

    cv2.imwrite(correspondant_img_path, corr_img)
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
        camera_points = np.asarray(camera_points).astype(np.float32)
        projector_points = np.asarray(projector_points).astype(np.float32)

        cam_norm = cv2.undistortPoints(src=camera_points, cameraMatrix=camera_K, distCoeffs=camera_d)
        proj_norm = cv2.undistortPoints(src=projector_points,  cameraMatrix=projector_K, distCoeffs=projector_d)

        # p1 = np.hstack((projector_K, np.zeros(shape=[3,1], dtype=np.float32))) * np.hstack((projector_R, projector_t))
        # p0 = np.hstack((camera_K, np.zeros(shape=[3,1], dtype=np.float32))) * np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])

        p1 = np.hstack((projector_R, projector_t))
        p0 = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])

        points_2d = cv2.triangulatePoints(p0, p1, cam_norm, proj_norm)
        points_2d = points_2d.T
        points_3d = cv2.convertPointsFromHomogeneous(points_2d)

        color_pts = []
        filter_3d = []
        for i in range(len(points_3d)):
            if 200 < points_3d[i][0][2] < 1400:
                filter_3d.append(points_3d[i])
                color_pts.append(cam_rgb[i])
        filter_3d = np.asarray(filter_3d)
        color_pts = np.asarray(color_pts)

        output_name_color = sys.argv[1] + "output_color.xyz"
        with open(output_name_color, "w") as f:
            for p, c in zip(filter_3d, color_pts):
                f.write("%d %d %d %d %d %d\n" % (p[0, 0], p[0, 1], p[0, 2], c[0, 0], c[0, 1], c[0, 2]))

        return filter_3d

        # mask = (points_3d[:, :, 2] > 200) & (points_3d[:, :, 2] < 1400)
        # return points_3d[mask]
        # return points_3d[mask]



def write_3d_points(points_3d):
    # ===== DO NOT CHANGE THIS FUNCTION =====

    print("write output point cloud")
    print(points_3d.shape)
    output_name = sys.argv[1] + "output.xyz"
    with open(output_name, "w") as f:
        for p in points_3d:
            f.write("%d %d %d\n" % (p[0, 0], p[0, 1], p[0, 2]))

    # return points_3d, camera_points, 2


if __name__ == '__main__':

    # ===== DO NOT CHANGE THIS FUNCTION =====

    # validate the input arguments
    if (len(sys.argv) != 2):
        help_message()
        sys.exit()

    points_3d = reconstruct_from_binary_patterns()
    write_3d_points(points_3d)
