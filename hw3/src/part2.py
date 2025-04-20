import numpy as np
import cv2
from cv2 import aruco
from tqdm import tqdm
from utils import solve_homography, warping


def planarAR(REF_IMAGE_PATH, VIDEO_PATH):
    video = cv2.VideoCapture(VIDEO_PATH)
    ref_image = cv2.imread(REF_IMAGE_PATH)
    h, w, c = ref_image.shape
    film_h, film_w = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    film_fps = video.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    videowriter = cv2.VideoWriter("output2.avi", fourcc, film_fps, (film_w, film_h))
    arucoDict = aruco.Dictionary_get(aruco.DICT_4X4_50)
    arucoParameters = aruco.DetectorParameters_create()
    ref_corns = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

    pbar = tqdm(total=int(video.get(cv2.CAP_PROP_FRAME_COUNT)))

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        corners, ids, _ = aruco.detectMarkers(frame, arucoDict, parameters=arucoParameters)
        if ids is not None and len(ids) >= 1:
            # Use the first detected marker (or filter by id)
            corner = corners[0][0].astype(np.float32)

            x_coords, y_coords = corner[:, 0], corner[:, 1]
            xmin, xmax = int(np.min(x_coords)), int(np.max(x_coords))
            ymin, ymax = int(np.min(y_coords)), int(np.max(y_coords))

            H = solve_homography(ref_corns, corner)

            frame = warping(ref_image, frame, H, ymin, ymax, xmin, xmax, direction='b')

        videowriter.write(frame)
        pbar.update(1)

    pbar.close()
    video.release()
    videowriter.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":

    # ================== Part 2: Marker-based planar AR========================
    VIDEO_PATH = '../resource/seq0.mp4'
    # TODO: you can change the reference image to whatever you want
    REF_IMAGE_PATH = '../resource/hehe.jpg' 
    planarAR(REF_IMAGE_PATH, VIDEO_PATH)