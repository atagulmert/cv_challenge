import cv2
import numpy as np
from matplotlib import pyplot as plt

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img1
        lines - corresponding epilines '''
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        cv2.line(img1, (x0,y0), (x1,y1), color,1)
        cv2.circle(img1,tuple(pt1), 10, color, -1)
        cv2.circle(img2,tuple(pt2), 10,color,-1)
    return img1,img2

def in_front_of_both_cameras(first_points, second_points, rot, trans):
    # check if the point correspondences are in front of both images
    score = 0
    rot_inv = rot
    for first, second in zip(first_points, second_points):
        first_z = np.dot(rot[0, :] - second[0]*rot[2, :], trans) / np.dot(rot[0, :] - second[0]*rot[2, :], second)
        first_3d_point = np.array([first[0] * first_z, second[0] * first_z, first_z])
        second_3d_point = np.dot(rot.T, first_3d_point) - np.dot(rot.T, trans)
        if first_3d_point[2] < 0 or second_3d_point[2] < 0:
            score -= 1
        else:
            score +=1
    return score
    
def camera_pose_estimation(img1,img2,K):
    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    index_params = dict(algorithm = 1, trees = 10)
    search_params = dict(checks=100)
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)

    pts1 = []
    pts2 = []

    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)


    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS,confidence=0.99)

    pts1 = pts1[mask.ravel()==1]
    pts2 = pts2[mask.ravel()==1]

    """
    # drawing lines on left image
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
    lines1 = lines1.reshape(-1,3)
    img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)

    # drawing lines on right image
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
    lines2 = lines2.reshape(-1,3)
    img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)
    """

    print("Fundamental matrix")
    print(F)

    # x'^T.F.x=0
    error_array = []
    for i in range(len(pts1)):
        pt1 = np.array([[pts1[i][0]], [pts1[i][1]], [1]])
        pt2 = np.array([[pts2[i][0], pts2[i][1], 1]])
        error = float(np.dot(np.dot(pt2,F),pt1))
        error_array.append(abs(error))
    
    print("Fundamental matrix error: ",np.mean(error_array))

    K_inv = np.linalg.inv(K)
    E = K.T.dot(F).dot(K)
    print("Essential Matrix:")
    print(E)

    U, S, V = np.linalg.svd(E)
    W = np.array([0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]).reshape(3, 3)
    if np.linalg.det(U):
        U = -U
    if np.linalg.det(V):
        V = -V
    R1 = U.dot(W).dot(V)
    R2 = U.dot(W.T).dot(V)
    T = U[:, 2]

    first_inliers = []
    second_inliers = []

    #R1, R2, T = cv2.decomposeEssentialMat(E)

    for i in range(len(pts1)):
        # normalize and homogenize the image coordinates
        first_inliers.append(K_inv.dot([pts1[i][0], pts1[i][1], 1.0]))
        second_inliers.append(K_inv.dot([pts2[i][0], pts2[i][1], 1.0]))
    # find correct pose out of four possible ones, check if a point is in front of both cameras
    # "that a reconstructed point X will be in front of both cameras in one of these four solutions only.
    # Thus, testing with a single point to determine if it is in front of both cameras is
    # sufficient to decide between the four different solutions for the camera matrix P0."
    # (Hartley Zisserman - 8.6.3 https://www.robots.ox.ac.uk/~vgg/hzbook/hzbook1/HZepipolar.pdf)
    R_list = [R1,R2]
    T_list = [T,-T]
    score_list = []
    for R in R_list:
        for T in T_list:
            score = in_front_of_both_cameras(first_inliers,second_inliers,R,T)
            score_list.append([[R,T],score])
    best_score = max(score_list,key=lambda x:x[1])
    R, T = best_score[0][0],best_score[0][1]

    # Converting rotation matrice to euler angles
    thetaX = np.arctan2(R[1][2], R[2][2])
    c2 = np.sqrt((R[0][0]*R[0][0] + R[0][1]*R[0][1]))

    thetaY = np.arctan2(-R[0][2], c2)

    s1 = np.sin(thetaX)
    c1 = np.cos(thetaX)

    thetaZ = np.arctan2((s1*R[2][0] - c1*R[1][0]), (c1*R[1][1] - s1*R[2][1]))
    print("Rotation matrix:")
    print(R)
    print("Pitch: %f, Yaw: %f, Roll: %f"%(thetaX*180/3.1415, thetaY*180/3.1415, thetaZ*180/3.1415))
    print("Translation vector:")
    print(T)

    #plt.subplot(121),plt.imshow(img5)
    #plt.subplot(122),plt.imshow(img3)
    #plt.show()

if __name__ == '__main__':
    K = np.float32([763, 0, 960, 0, 763, 540, 0, 0, 1]).reshape(3,3)
    img1 = cv2.imread('data/img1.png',0)
    img2 = cv2.imread('data/img2.png',0)
    img3 = cv2.imread('data/img3.png',0)
    print("******************IMG1->IMG2*********************")
    camera_pose_estimation(img1,img2,K)
    print("******************IMG1->IMG3*********************")
    camera_pose_estimation(img1,img3,K)