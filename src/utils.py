import cv2
import numpy as np

def track_candidates(C_old, F_old, Tau_old, img1, img2):
    # Parameters for the KLT (Lucas-Kanade) tracker
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    # Calculate the optical flow from img1 to img2
    C_new, st, err = cv2.calcOpticalFlowPyrLK(img1, img2, C_old, None, **lk_params)

    C_new = C_new.T
    C_new.reshape(2, C_new.shape[2])
    st = st.flatten()
    # Select good points for tracking from 00 to 01
    good_old_01 = C_old[:][st == 1]
    good_new_01 = C_new[:][st == 1]
    F_old = F_old[:][st == 1]
    Tau_old = Tau_old[:][st == 1]
    return good_new_01, F_old, Tau_old

def triangulate_ransac_pnp(X_old, P_new, K):
    # Triangulate the points using the PnP algorithm
    _, R_vec, t, inliers = cv2.solvePnPRansac(X_old, P_new, K, None)
    R, _ = cv2.Rodrigues(R_vec)
    inliers = inliers.reshape(-1)
    X_old = X_old[inliers]
    P_new = P_new[inliers]
    return X_old, P_new, R, t


def expand_C(C, F, Tau, img, R, t):
    # Extract Harris corners from the image
    C_new = cv2.goodFeaturesToTrack(img, maxCorners=200, qualityLevel=0.01, minDistance=7, blockSize=7)
    
    # Add new non overlapping corners to the existing corners
    C = np.concatenate((C, C_new), axis=1)
    F = np.concatenate((F, C_new), axis=1)
    
    # Add the pose to the new ones
    T = np.concatenate((R, t), axis=1)
    Tau = np.concatenate((Tau, np.full(C_new.shape[1], T)), axis=1)
    
    # Remove duplicates
    C = np.unique(C, axis=1) #TODO: maybe there is a better way to do this)
    F = np.unique(F, axis=1)
    
    return C, F, Tau
    


def get_new_candidate_points(image, R, t):
    C_new = cv2.goodFeaturesToTrack(image, maxCorners=200, qualityLevel=0.01, minDistance=7, blockSize=7)
    T = np.concatenate((R, t), axis=1)
    Tau = np.tile(T, (C_new.shape[0], 1, 1))
    F = C_new

    return C_new, F, Tau
