import cv2 as cv
import numpy as np
import json
import os
import shutil
from .Singleton import Singleton
import time
from scipy import linalg, optimize, signal
from scipy.spatial.transform import Rotation
import copy
from typing import List, Tuple
from colorama import Fore, Style

@Singleton
class Cameras:
    def __init__(self):
        # camera-parameters: intrinsic matirx, distortion & rotation
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, "camera-params.json")
        with open(filename) as f: # use "with" to automatically close json file
            self.camera_params = json.load(f)
        
        self.cam_index = [4, 6]
        self.num_cameras = len(self.cam_index)
        self.width,self.height,self.fps = 640, 480, 120

        self.cameras = [cv.VideoCapture(i) for i in self.cam_index]
        for idx, camera in enumerate(self.cameras):
            if not camera.isOpened():
                raise RuntimeError(f"Cannot open camera {self.cam_index[idx]}")
            
            camera.set(cv.CAP_PROP_FRAME_WIDTH, self.width)
            camera.set(cv.CAP_PROP_FRAME_HEIGHT, self.height)
            camera.set(cv.CAP_PROP_FPS, self.fps)

            CAP_PROP_FRAME_WIDTH = camera.get(cv.CAP_PROP_FRAME_WIDTH)
            CAP_PROP_FRAME_HEIGHT = camera.get(cv.CAP_PROP_FRAME_HEIGHT)
            CAP_PROP_FPS = camera.get(cv.CAP_PROP_FPS)

            print(f"\nReal width: {CAP_PROP_FRAME_WIDTH}")
            print(f"Real height: {CAP_PROP_FRAME_HEIGHT}")
            print(f"Real fps: {CAP_PROP_FPS}")
        input("\nCamera initialization is completed, press enter to continue.")

        self.is_calculating_object_pose = False

    def camera_read(self):
        frames = []
        for camera in self.cameras:
            ret, frame = camera.read()
            if not ret:
                raise RuntimeError("Error: camera.read return=false")
            frames.append(frame)
        
        for i in range(self.num_cameras):
            # rotate the frame (seems all "rotation" is zero)
            frames[i] = np.rot90(frames[i], k=self.camera_params[i]["rotation"])
            # make the frame equal width & height
            #frames[i] = make_square(frames[i])
            # undistort the frame
            frames[i] = cv.undistort(frames[i], self.get_camera_params(i)["intrinsic_matrix"], self.get_camera_params(i)["distortion_coef"])
            # smooths the frame using a Gaussian filter
            # frames[i] = cv.GaussianBlur(frames[i],(9,9),0)
            '''
            # enhances specific features in the frame, like bright point
            kernel = np.array([[-2,-1,-1,-1,-2],
                                [-1,1,3,1,-1],
                                [-1,3,4,3,-1],
                                [-1,1,3,1,-1],
                                [-2,-1,-1,-1,-2]])
            frames[i] = cv.filter2D(frames[i], -1, kernel)
            '''
            # convert the frame from RGB to BGR for OpenCV compatibility
            # frames[i] = cv.cvtColor(frames[i], cv.COLOR_RGB2BGR)

        return frames

    def find_dot(self, img):
        # img = cv.GaussianBlur(img,(5,5),0)

        # convert the frame from RGB to GREY
        grey = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        # thretholding converts GREY image into binary image
        grey = cv.threshold(grey, 200, 255, cv.THRESH_BINARY)[1]
        # find the white parts/contours/white dots
        contours,_ = cv.findContours(grey, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        # draw the detected contours on the origin image in green
        #img = cv.drawContours(img, contours, -1, (0,255,0), 1)

        # initialize an empty list to store dot coordinates(x, y)
        image_points = []
        # iterate over each contour
        for contour in contours:
            moments = cv.moments(contour) # computer the moments of the contour
            if moments["m00"] != 0: # "m00" is the total "mass" of the contour
                center_x = int(moments["m10"] / moments["m00"])
                center_y = int(moments["m01"] / moments["m00"]) # like physical mass point
                # overlay (x, y) on the image
                cv.putText(img, f'({center_x}, {center_y})', (center_x,center_y - 15), cv.FONT_HERSHEY_SIMPLEX, 0.3, (100,255,100), 1)
                # mark the centriod with a small circle
                cv.circle(img, (center_x,center_y), 1, (100,255,100), -1)
                # store the centriod coordinates(x, y)
                image_points.append([center_x, center_y])
        # handle the case where no dots are found
        if len(image_points) == 0:
            image_points = [[None, None]]
        # return the processed image and the list of detected points' coordinates(x, y)
        return img, image_points

    def get_frames(self):
        frames = self._camera_read()
        return np.hstack(frames)

    def get_camera_params(self, cam_index):
        return {
            "intrinsic_matrix": np.array(self.camera_params[cam_index]["intrinsic_matrix"]),
            "distortion_coef": np.array(self.camera_params[cam_index]["distortion_coef"]),
            "rotation": self.camera_params[cam_index]["rotation"]
        }

    def set_camera_params(self, cam_index, intrinsic_matrix=None, distortion_coef=None):
        if intrinsic_matrix is not None:
            self.camera_params[cam_index]["intrinsic_matrix"] = intrinsic_matrix
        
        if distortion_coef is not None:
            self.camera_params[cam_index]["distortion_coef"] = distortion_coef

    def start_calculating_object_pose(self):
        self.is_calculating_object_pose = True
    
    def stop_calculating_object_pose(self):
        self.is_calculating_object_pose = False

"""
Brief: draw lines in image
Params:
    image: np.array opencv
    lines: py.list 2D
Retval: image with lines
"""
def drawlines(image, lines):
    row, column = image.shape[:2]
    for l in lines:
        color = tuple(np.random.randint(0,255,3).tolist()) # generate color randomly
        # case by case calculate the boundary 
        if l[0]==0:
            y0 = y1 = int(-l[2]/l[1])
            x0, x1 = 0, column
        elif l[1] == 0:
            x0=x1 = int(-l[2]/l[0])
            y0, y1 = 0, row
        else:
            # Calculate the first point (x0, y0)
            x0 = int(-l[2]/l[0]) if 0<=-l[2]/l[0]<column else column
            y0 = 0 if 0<=-l[2]/l[0]<column else int(-(l[2]+l[0]*x0)/l[1])
            # Calculate the second point (x1, y1)
            y1 = int(-l[2]/l[1]) if 0<=-l[2]/l[1]<row else row
            x1 = 0 if 0 <=-l[2]/l[1]<row else int(-(l[2]+l[1]*y1)/l[0])
    image = cv.line(image, (x0,y0), (x1,y1), color, 1)
    return image

"""
Brief: triangulate single world point,
        through image points' coordinates
        &camera poses,
        using "DLT" method
Params: 
    image_points: points' coordinates from cameras(<=num_cameras), 
                    but same point in real world
    camera_poses: cameras' poses array
Retval: object_point
"""
def triangulate_point(image_points, camera_poses):
    image_points = np.array(image_points) # python list->numpy array
    cameras = Cameras.instance() # get camera parameter
    none_indicies = np.where(np.all(image_points == None, axis=1))[0] # find out [None, None]'s index
    image_points = np.delete(image_points, none_indicies, axis=0)
    camera_poses = np.delete(camera_poses, none_indicies, axis=0)

    if len(image_points) <= 1:
        return [None, None, None] # cannot triangulate point

    Ps = [] # store cameras' projection matrix

    for i, camera_pose in enumerate(camera_poses):
        Rt = np.c_[camera_pose["R"], camera_pose["t"]]
        P = np.array(cameras.camera_params[i]["intrinsic_matrix"]) @ Rt
        Ps.append(P)

    # https://temugeb.github.io/computer_vision/2021/02/06/direct-linear-transorms.html
    # https://www.bilibili.com/video/BV1ExWxesEVf/?spm_id_from=333.1007.top_right_bar_window_history.content.click
    
    # Direct Linear Transform
    def DLT(Ps, image_points):
        A = []

        for P, image_point in zip(Ps, image_points):
            A.append(image_point[1]*P[2,:] - P[1,:])
            A.append(P[0,:] - image_point[0]*P[2,:])
        
        A = np.array(A).reshape((len(Ps)*2, 4))
        _, _, Vh = linalg.svd(A)
        # U, s, Vh is not unique, Vh[3, 3]>0 is required
        if Vh[3, 3] < 0:
            Vh = -Vh
        '''
        # original code
        B = A.transpose() @ A
        U, s, Vh = linalg.svd(B, full_matrices = False)
        '''
        
        object_point = Vh[3,0:3] / Vh[3,3]
        
        return object_point
    
    object_point = DLT(Ps, image_points)

    return object_point

"""
Brief: trangulate several world points,
        through image points' coordinates,
        &camera poses,
        based on "trangulate point"
Params: 
    image_points: points' coordinates from cameras,
                    image_points[i] is equal to image_points in "triangulate_point"
    camera_poses: camera poses array
Retval: object points
"""
def triangulate_points(image_points, camera_poses):
    object_points = []
    for image_point_i in image_points:
        object_point = triangulate_point(image_point_i, camera_poses)
        object_points.append(object_point)

    return np.array(object_points)

"""
Brief: reproject the object point calculated by "triangulate_point",
        do some calculation and return reprojection error
Params: 
    image_points: one of "triangulate_point"'s params
    camera_poses: one of "triangulate_point"'s params
    object_point: retval of "triangulate_point"
Retval: 
    reprojection_error: the mean squared distance between the object point and reproject point for each camera 
"""
def calculate_reprojection_error(image_points, object_point, camera_poses):
    cameras = Cameras.instance()
    # delete invalid data
    image_points = np.array(image_points)
    none_indicies = np.where(np.all(image_points == None, axis=1))[0]
    image_points = np.delete(image_points, none_indicies, axis=0)
    camera_poses = np.delete(camera_poses, none_indicies, axis=0)

    if len(image_points) <= 1:
        return None
    
    image_points_t = image_points.transpose((0,1)) # seems meaningless
    # calculate reprojection error
    errors = np.array([])
    for i, camera_pose in enumerate(camera_poses):
        if np.all(image_points[i] == None, axis=0):
            continue
        projected_img_points, _ = cv.projectPoints(
            np.expand_dims(object_point, axis=0).astype(np.float32),
            np.array(camera_pose["R"], dtype=np.float64),
            np.array(camera_pose["t"], dtype=np.float64),
            cameras.get_camera_params(i)["intrinsic_matrix"],
            np.array([]) # meant to be distort coef
        )
        projected_img_point = projected_img_points[:,0,:][0]
        errors = np.concatenate([errors, (image_points_t[i]-projected_img_point).flatten() ** 2])
    return errors.mean()

"""
Brief: reproject the object points calculated by "triangulate_points",
        do some calculation and return reprojection error
Params: 
    image_points: one of "triangulate_points"' params
    camera_poses: one of "triangulate_points"' params
    object_point: retval of "triangulate_point"
Retval: 
    reprojection_errors: array of reprojection error
"""
def calculate_reprojection_errors(image_points, object_points, camera_poses):
    errors = np.array([])
    for image_points_i, object_point in zip(image_points, object_points):
        error = calculate_reprojection_error(image_points_i, object_point, camera_poses)
        if error is None:
            continue
        errors = np.concatenate([errors, [error]])

    return errors

"""
Brief: after given a rough estimate from "epipolar geometry",
        use the image_points and the initial camera_poses,
        run a nonlinear optimizer from scipy,
        (params: camera_poses, residual: reprojection_errors)
        return the camera_poses that's optimized
Params: 
    image_points: image_points in "triangulate_points" function
    camera_poses: camera_poses in "triangulate_points" function
    socketio: used to communicate with frontend, but temporarily not used
Retval: camera_poses that's optimized
"""
def bundle_adjustment(image_points, camera_poses):
    cameras = Cameras.instance()

    def params_to_camera_poses(params):
        # camera1: as relative/world coordinate
        focal_distances = []
        num_cameras = int((params.size-1)/7)+1
        camera_poses = [{
            "R": np.eye(3),
            "t": np.array([0,0,0], dtype=np.float32)
        }]
        focal_distances.append(params[0])
        # camera2,3,4
        for i in range(0, num_cameras-1):
            focal_distances.append(params[i*7+1])
            camera_poses.append({
                "R": Rotation.as_matrix(Rotation.from_rotvec(params[i*7+2 : i*7+3+2])),
                "t": params[i*7+3+2 : i*7+6+2]
            })

        return camera_poses, focal_distances

    def residual_function(params):
        camera_poses, focal_distances = params_to_camera_poses(params)
        for i in range(0, len(camera_poses)):
            intrinsic = cameras.get_camera_params(i)["intrinsic_matrix"]
            intrinsic[0, 0] = focal_distances[i]
            intrinsic[1, 1] = focal_distances[i]
            # cameras.set_camera_params(i, intrinsic)
        object_points = triangulate_points(image_points, camera_poses)
        errors = calculate_reprojection_errors(image_points, object_points, camera_poses)
        errors = errors.astype(np.float32)

        return errors

    focal_distance = cameras.get_camera_params(0)["intrinsic_matrix"][0, 0]
    init_params = np.array([focal_distance])
    for i, camera_pose in enumerate(camera_poses[1:]):
        rot_vec = Rotation.as_rotvec(Rotation.from_matrix(camera_pose["R"])).flatten()
        focal_distance = cameras.get_camera_params(i)["intrinsic_matrix"][0,0]
        init_params = np.concatenate([init_params, [focal_distance]])
        init_params = np.concatenate([init_params, rot_vec])
        init_params = np.concatenate([init_params, camera_pose["t"].flatten()])

    res = optimize.least_squares(
        residual_function, init_params, verbose=2, loss="cauchy", ftol=1E-4
    )

    return params_to_camera_poses(res.x)[0]

"""
Brief: find points' correspondances and object points
Params: 
    image_points: sorted by camera index rather than correspondance
    camera_poses: cameras' poses in "triangulate_points" function
    frames: several frames of cameras at one moment
Retval: 
    np.array(errors): errors related to object points
    np.array(object_points): object points in different correspondances
    frames: frames that've been drawn
"""
def find_point_correspondance_and_object_points(image_points, camera_poses, frames):
    cameras = Cameras.instance()
    # clear the [None, None] part for each image_points_i
    for image_point_i in image_points:
        try:
            image_point_i.remove([None, None])
        except:
            pass
    # camera0 is chosen to be the baseline
    # correspondances' structure is shown in the tablet
    correspondances = [[[i]] for i in image_points[0]]

    Ps = []
    for i, camera_pose in enumerate(camera_poses):
        Rt = np.c_[camera_pose["R"], camera_pose["t"]]
        P = cameras.camera_params[i]["intrinsic_matrix"] @ Rt
        Ps.append(P)
    
    root_image_points = [{"camera": 0, "point": point} for point in image_points[0]]

    for i in range(1, len(camera_poses)):
        epipolar_lines = []
        # calculate epipolar lines and draw lines in frame
        for root_image_point in root_image_points:
            F = fundamental_from_projections(Ps[root_image_point["camera"]], Ps[i])
            line = cv.computeCorrespondEpilines(np.array([root_image_point["point"]], dtype=np.float32), 1, F) # return shape (1,1,3)
            epipolar_lines.append(line[0,0].tolist())
            frames[i] = drawlines(frames[i], line[0])
        # initialize the not "closet match" image points
        not_closest_match_image_points = np.array(image_points[i])
        points = np.array(image_points[i])

        for j, [a,b,c] in enumerate(epipolar_lines):
            distances_to_line = np.array([])
            if len(points) != 0:
                distances_to_line = np.abs(a*points[:,0] + b*points[:,1] + c) / np.sqrt(a**2+b**2)
            
            possible_matches = points[distances_to_line<5].copy() # possibly not only one
            # There's a comment in source code

            # sort possible matches from smallest to largest
            distances_to_line = distances_to_line[distances_to_line<5]
            possible_matches_sorter = distances_to_line.argsort()
            possible_matches = possible_matches[possible_matches_sorter]

            if len(possible_matches) == 0:
                for possible_group in correspondances[j]:
                    possible_group.append([None, None])
            else:
                not_closest_match_image_points = [row for row in not_closest_match_image_points.tolist() if row != possible_matches.tolist()[0]]
                not_closest_match_image_points = np.array(not_closest_match_image_points)
                
                new_correspondances_j = []
                for possible_match in possible_matches:
                    temp = copy.deepcopy(correspondances[j])
                    for possible_group in temp:
                        possible_group.append(possible_match.tolist())
                    new_correspondances_j += temp
                correspondances[j] = new_correspondances_j
        
        for not_closest_match_image_point in not_closest_match_image_points:
            root_image_points.append({"camera": i, "point": not_closest_match_image_point})
            temp = [[[None, None]] * i]
            temp[0].append(not_closest_match_image_point.tolist())
            correspondances.append(temp)
    
    object_points = []
    errors = []
    for image_points in correspondances:
        object_points_i = triangulate_points(image_points, camera_poses)

        if np.all(object_points_i == None):
            continue

        errors_i = calculate_reprojection_errors(image_points, object_points_i, camera_poses)

        object_points.append(object_points_i[np.argmin(errors_i)])
        errors.append(np.min(errors_i))
    
    return np.array(errors), np.array(object_points), frames

"""
Brief: Calculate object pose using "centroid and distance method"
Params: 
    object_points: object_points in world coordinate frame
    threold: related to the positioning accuracy
Retval: 
    R:
    t:
    distances:
"""
def calculate_object_pose(object_points: np.ndarray, threold) -> np.ndarray:
    R,t,distances = np.array([[]]), np.array([]), np.array([])
    cameras = Cameras.instance()
    if cameras.is_calculating_object_pose == True:  
        if len(object_points) > 2:
            centroid = np.array([
                object_points[:,0].mean(),
                object_points[:,1].mean(),
                object_points[:,2].mean()
            ])
            distances = np.linalg.norm(object_points - centroid, axis=1)

            min_distance = None
            max_distance = None
            sorted_distances = np.sort(distances)
            for i in range(len(distances)):
                if i == len(distances)-1:
                    print("No min_distance.")
                    break
                elif sorted_distances[i+1] - sorted_distances[i] > threold:
                    min_distance = sorted_distances[i]
                    break
                else:
                    continue

            for i in range(len(distances)):
                if i == len(distances)-1:
                    print("No max_distance.")
                    break
                elif sorted_distances[len(distances)-i-1] - sorted_distances[len(distances)-i-2] > threold:
                    max_distance = sorted_distances[len(distances)-i-1]
                    break
                else:
                    continue
            if min_distance != None and max_distance != None:
                min_index = np.where(distances == min_distance)[0][0]
                max_index = np.where(distances == max_distance)[0][0]

                x_axis = object_points[min_index] - centroid
                y_axis = object_points[max_index] - centroid
                z_axis = np.cross(x_axis, y_axis)
                y_axis = np.cross(z_axis, x_axis)

                x_axis = x_axis / linalg.norm(x_axis)
                y_axis = y_axis / linalg.norm(y_axis)
                z_axis = z_axis / linalg.norm(z_axis)

                R = np.array([x_axis, y_axis, z_axis])
                t = np.dot(R,-centroid).reshape(3,1)
            else:
                print("Shape error.")
        else:
            print("Insufficient points to calculate pose.")

    return R, t, distances


"""
Below are functions built from opencv contrib module:sfm ,
complie sfm from source is so fucking painful
"""
def essential_from_fundamental(F: np.ndarray, K1: np.ndarray, K2: np.ndarray) -> np.ndarray:
    """
    Calculate the essential matrix from the fundamental matrix (F) and camera matrices (K1, K2).

    Adapted and modified from OpenCV's essentialFromFundamental function in the opencv_sfm module
    """

    assert F.shape == (3, 3), "F must be a 3x3 matrix"
    assert K1.shape == (3, 3), "K1 must be a 3x3 matrix"
    assert K2.shape == (3, 3), "K2 must be a 3x3 matrix"

    E = np.dot(np.dot(K2.T, F), K1)
    return E

def fundamental_from_projections(P1: np.ndarray, P2: np.ndarray) -> np.ndarray:
    """
    Calculate the fundamental matrix from the projection matrices (P1, P2).
    
    Adapted and modified from OpenCV's fundamentalFromProjections function in the opencv_sfm module
    """

    assert P1.shape == (3, 4), "P1 must be a 3x4 matrix"
    assert P2.shape == (3, 4), "P2 must be a 3x4 matrix"

    F = np.zeros((3, 3))

    X = np.array([
        np.vstack((P1[1, :], P1[2, :])),
        np.vstack((P1[2, :], P1[0, :])),
        np.vstack((P1[0, :], P1[1, :]))
    ])

    Y = np.array([
        np.vstack((P2[1, :], P2[2, :])),
        np.vstack((P2[2, :], P2[0, :])),
        np.vstack((P2[0, :], P2[1, :]))
    ])

    for i in range(3):
        for j in range(3):
            XY = np.vstack((X[j], Y[i]))
            F[i, j] = np.linalg.det(XY)

    return F

def motion_from_essential(E: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Calculate the possible rotations and translations from the essential matrix (E).

    Adapted and modified from OpenCV's motionFromEssential function in the opencv_sfm module
    """
    assert E.shape == (3, 3), "Essential matrix must be 3x3."

    R1, R2, t = cv.decomposeEssentialMat(E)

    rotations_matrices = [R1, R1, R2, R2]
    translations = [t, -t, t, -t]

    return rotations_matrices, translations
