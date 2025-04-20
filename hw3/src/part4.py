import numpy as np
import cv2
import random
from tqdm import tqdm
from utils import solve_homography, warping

random.seed(999)
# Step 2: 自行實作 RANSAC
def compute_ransac_homography(src_pts, dst_pts, max_iter=4500, threshold=1.0):
    best_H = None
    max_inliers = 0
    inliers = []

    N = src_pts.shape[0]
    
    # RANSAC loop
    for _ in range(max_iter):
        # Randomly select 4 points
        indices = np.random.choice(N, 4, replace=False)
        src_subset = src_pts[indices]
        dst_subset = dst_pts[indices]
        
        # Compute homography for the current subset
        H_candidate = solve_homography(src_subset, dst_subset)
        
        # Check for valid homography
        if H_candidate is None:
            continue
        
        # Step 3: Find inliers by checking distance to the transformed points
        projected_pts = cv2.perspectiveTransform(src_pts.reshape(-1, 1, 2), H_candidate)
        
        # Flatten the projected points to shape (N, 2)
        projected_pts = projected_pts.reshape(-1, 2)
        
        # Calculate the distance between the projected points and actual destination points
        distances = np.linalg.norm(projected_pts - dst_pts, axis=1)
        
        # Inliers are points with a distance smaller than the threshold
        current_inliers = distances < threshold
        num_inliers = np.sum(current_inliers)
        
        # If we have more inliers, update the best homography
        if num_inliers > max_inliers:
            best_H = H_candidate
            max_inliers = num_inliers
            inliers = current_inliers

    return best_H, inliers

def panorama(imgs):
    h_img, w_img, c = imgs[0].shape
    h_max = max([img.shape[0] for img in imgs])
    w_max = sum([img.shape[1] for img in imgs])
    dst = np.zeros((h_max, w_max, c), dtype=np.uint8)

    last_best_H = [np.eye(3)]

    for idx in tqdm(range(len(imgs) - 1)):
        im1 = imgs[idx]
        im2 = imgs[idx + 1]

        # 1. Feature detection and matching
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(im1, None)
        kp2, des2 = orb.detectAndCompute(im2, None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

        # 2. Find homography with RANSAC
        H, inliers = compute_ransac_homography(src_pts, dst_pts)

        # 3. Chain the homographies
        last_best_H.append(last_best_H[-1] @ np.linalg.inv(H))

    # 4. Apply warping for each image
    for i in range(len(imgs)):
        H = last_best_H[i]
        dst = warping(imgs[i], dst, H, 0, h_max, 0, w_max, direction='b')

    return dst

if __name__ == "__main__":
    # ================== Part 4: Panorama ========================
    # TODO: change the number of frames to be stitched
    FRAME_NUM = 3
    imgs = [cv2.imread('../resource/frame{:d}.jpg'.format(x)) for x in range(1, FRAME_NUM + 1)]
    output4 = panorama(imgs)
    cv2.imwrite('output4.png', output4)