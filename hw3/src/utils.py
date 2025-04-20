import numpy as np

def solve_homography(u, v):
    """
    This function should return a 3-by-3 homography matrix,
    u, v are N-by-2 matrices, representing N corresponding points for v = T(u)
    :param u: N-by-2 source pixel location matrices
    :param v: N-by-2 destination pixel location matrices
    :return:
    """
    N = u.shape[0]
    H = None

    if v.shape[0] is not N:
        print('u and v should have the same size')
        return None
    if N < 4:
        print('At least 4 points should be given')

    # Step 1: Form matrix A such that A * h = 0
    A = []
    for i in range(N):
        x, y = u[i][0], u[i][1]
        x_prime, y_prime = v[i][0], v[i][1]

        A.append([-x, -y, -1,  0,  0,  0, x * x_prime, y * x_prime, x_prime])
        A.append([ 0,  0,  0, -x, -y, -1, x * y_prime, y * y_prime, y_prime])
    
    A = np.array(A)

    # Step 2: Solve Ah = 0 using SVD
    U, S, Vt = np.linalg.svd(A)
    h = Vt[-1]  # the last row of Vt (smallest singular value)
    H = h.reshape((3, 3))  # reshape h to 3x3 matrix

    # Normalize so that H[2, 2] = 1
    H = H / H[2, 2]

    return H


def warping(src, dst, H, ymin, ymax, xmin, xmax, direction='f'):
    """
    Perform forward/backward warpping without for loops. i.e.
    for all pixels in src(xmin~xmax, ymin~ymax),  warp to destination
          (xmin=0,ymin=0)  source                       destination
                         |--------|              |------------------------|
                         |        |              |                        |
                         |        |     warp     |                        |
    forward warp         |        |  --------->  |                        |
                         |        |              |                        |
                         |--------|              |------------------------|
                                 (xmax=w,ymax=h)

    for all pixels in dst(xmin~xmax, ymin~ymax),  sample from source
                            source                       destination
                         |--------|              |------------------------|
                         |        |              | (xmin,ymin)            |
                         |        |     warp     |           |--|         |
    backward warp        |        |  <---------  |           |__|         |
                         |        |              |             (xmax,ymax)|
                         |--------|              |------------------------|

    :param src: source image
    :param dst: destination output image
    :param H:
    :param ymin: lower vertical bound of the destination(source, if forward warp) pixel coordinate
    :param ymax: upper vertical bound of the destination(source, if forward warp) pixel coordinate
    :param xmin: lower horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param xmax: upper horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param direction: indicates backward warping or forward warping
    :return: destination output image
    """

    h_src, w_src, ch = src.shape
    h_dst, w_dst, ch = dst.shape

    H_inv = np.linalg.inv(H)

    # TODO: 1.meshgrid the (x,y) coordinate pairs
    x, y = np.meshgrid(np.arange(xmin, xmax), np.arange(ymin, ymax))

    # TODO: 2.reshape the destination pixels as N x 3 homogeneous coordinate
    xy_homog = np.stack([x.ravel(), y.ravel(), np.ones_like(x).ravel()], axis=0)  # shape: (3, N)

    if direction == 'b':
        # TODO: 3.apply H_inv to the destination pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        src_homog = np.dot(H_inv, xy_homog)  # Apply inverse homography
        src_coords = src_homog[:2] / src_homog[2:3]  # Convert from homogeneous coordinates
        u = src_coords[0].reshape((ymax-ymin), (xmax-xmin))  # Reshape to match original grid
        v = src_coords[1].reshape((ymax-ymin), (xmax-xmin))

        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of source image)
        mask = (u >= 0) & (u < w_src) & (v >= 0) & (v < h_src)

        # TODO: 5.sample the source image with the masked and reshaped transformed coordinates
        # This requires bilinear interpolation for proper sampling
        u_safe = np.clip(u, 0, w_src-1)
        v_safe = np.clip(v, 0, h_src-1)
        
        # Get the four surrounding pixel coordinates
        u0 = np.floor(u_safe).astype(np.int32)
        u1 = np.ceil(u_safe).astype(np.int32)
        v0 = np.floor(v_safe).astype(np.int32)
        v1 = np.ceil(v_safe).astype(np.int32)
        
        # Calculate interpolation weights
        w_u1 = u_safe - u0
        w_u0 = 1 - w_u1
        w_v1 = v_safe - v0
        w_v0 = 1 - w_v1
        
        # Sample the four surrounding pixels
        img_00 = src[v0, u0]
        img_01 = src[v0, u1]
        img_10 = src[v1, u0]
        img_11 = src[v1, u1]
        
        # Perform bilinear interpolation
        sampled = (w_u0.reshape(-1, 1) * w_v0.reshape(-1, 1)) * img_00.reshape(-1, ch) + \
                  (w_u1.reshape(-1, 1) * w_v0.reshape(-1, 1)) * img_01.reshape(-1, ch) + \
                  (w_u0.reshape(-1, 1) * w_v1.reshape(-1, 1)) * img_10.reshape(-1, ch) + \
                  (w_u1.reshape(-1, 1) * w_v1.reshape(-1, 1)) * img_11.reshape(-1, ch)
        
        sampled = sampled.reshape(ymax-ymin, xmax-xmin, ch)

        # TODO: 6. assign to destination image with proper masking
        dst[ymin:ymax, xmin:xmax][mask] = sampled[mask]

    elif direction == 'f':
        # TODO: 3.apply H to the source pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        dst_homog = np.dot(H, xy_homog)  # Apply homography
        dst_coords = dst_homog[:2] / dst_homog[2:3]  # Convert from homogeneous coordinates
        u = dst_coords[0].reshape((ymax-ymin), (xmax-xmin))  # Reshape to match original grid
        v = dst_coords[1].reshape((ymax-ymin), (xmax-xmin))

        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of destination image)
        mask = (u >= 0) & (u < w_dst) & (v >= 0) & (v < h_dst)

        # TODO: 5.filter the valid coordinates using previous obtained mask
        v_valid = v[mask].astype(np.int32)
        u_valid = u[mask].astype(np.int32)
        
        # Get corresponding source coordinates
        y_valid = y[mask]
        x_valid = x[mask]

        # TODO: 6. assign to destination image using advanced array indicing
        dst[v_valid, u_valid] = src[y_valid, x_valid]

    return dst
