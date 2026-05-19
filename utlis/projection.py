import numpy as np
import scipy.io as sio


def project_to_2d(
    pts: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray
) -> np.ndarray:
    """Project 3D points to 2D using the project MATLAB convention."""
    M = np.concatenate((R, t), axis=0) @ K
    projPts = np.concatenate((pts, np.ones((pts.shape[0], 1))), axis=1) @ M
    projPts[:, :2] = projPts[:, :2] / projPts[:, 2:]

    return projPts


def distortPoints(points, intrinsicMatrix, radialDistortion, tangentialDistortion):
    """Distort points according to camera parameters.

    Ported from Matlab 2018a.
    """
    cx = intrinsicMatrix[2, 0]
    cy = intrinsicMatrix[2, 1]
    fx = intrinsicMatrix[0, 0]
    fy = intrinsicMatrix[1, 1]
    skew = intrinsicMatrix[1, 0]

    center = np.array([cx, cy])
    centeredPoints = points - center[np.newaxis, :]

    yNorm = centeredPoints[:, 1] / fy
    xNorm = (centeredPoints[:, 0] - skew * yNorm) / fx

    r2 = xNorm**2 + yNorm**2
    r4 = r2 * r2
    r6 = r2 * r4

    k = np.zeros((3,))
    k[:2] = radialDistortion[:2]
    if len(radialDistortion) < 3:
        k[2] = 0
    else:
        k[2] = radialDistortion[2]
    alpha = k[0] * r2 + k[1] * r4 + k[2] * r6

    p = tangentialDistortion
    xyProduct = xNorm * yNorm
    dxTangential = 2 * p[0] * xyProduct + p[1] * (r2 + 2 * xNorm**2)
    dyTangential = p[0] * (r2 + 2 * yNorm**2) + 2 * p[1] * xyProduct

    normalizedPoints = np.stack((xNorm, yNorm)).T
    distortedNormalizedPoints = (
        normalizedPoints
        + normalizedPoints * np.array([alpha, alpha]).T
        + np.stack((dxTangential, dyTangential)).T
    )

    distortedPointsX = (
        (distortedNormalizedPoints[:, 0] * fx)
        + cx
        + (skew * distortedNormalizedPoints[:, 1])
    )
    distortedPointsY = distortedNormalizedPoints[:, 1] * fy + cy
    distortedPoints = np.stack((distortedPointsX, distortedPointsY))

    return distortedPoints


def load_cameras(path):
    d = sio.loadmat(path)
    fns = ["K", "RDistort", "TDistort", "r", "t"]
    camnames = [cam[0] for cam in d["camnames"][0]]
    cam_params = d["params"]

    cameras = {}
    for i, camname in enumerate(camnames):
        cameras[camname] = {}
        for j, fn in enumerate(fns):
            cameras[camname][fn] = cam_params[i][0][0][0][j]

    return cameras
