from mocap.helpers import *
"""
Below code is running with 2 cameras
"""

if __name__ == "__main__":
    folder_path = "extrinsic"

    ## Delete cache
    shutil.rmtree(folder_path, ignore_errors=True)
    os.makedirs(folder_path, exist_ok=True)

    ## Choose cameras and modify number of point pairs
    num_image_point_pairs = 30
    camera_indices = [0,1]

    ## Read frames and collect image points
    cameras = Cameras.instance()
    cv.waitKey(50)

    image_point_pairs = [[],[]] # don't forget to convert to np.array
    for _, cam_idx in enumerate(camera_indices):
        os.makedirs(f"{folder_path}/camera_{cam_idx}", exist_ok=True)

    for i in range(num_image_point_pairs):
        input(f"\nPress enter to capture number{i} image")
        frames = cameras.camera_read()
        image_point_0 = cameras.find_dot(frames[0])[1]
        image_point_1 = cameras.find_dot(frames[1])[1]
        print(image_point_0, image_point_1)
        if len(image_point_0)==1 and len(image_point_1)==1 and image_point_0[0][0] is not None and image_point_1[0][0] is not None and i>4:
            image_point_pairs[0].append(image_point_0[0])
            image_point_pairs[1].append(image_point_1[0])
            filename0 = f"{folder_path}/camera_{camera_indices[0]}/image_{i}.jpg"
            filename1 = f"{folder_path}/camera_{camera_indices[1]}/image_{i}.jpg"
            cv.imwrite(filename0, frames[0])
            cv.imwrite(filename1, frames[1])
            print("Capture sucess, move the marker and continue.")
        else:
            print("Capture failed, move the marker and continue.")
            continue

    image_point_pairs = np.array(image_point_pairs)
    print(f"\nimage_point_pairs:\n{image_point_pairs}")

    if len(image_point_pairs[0]) <= 8:
        raise RuntimeError(f"image point pairs <= 8")
    else:
        print("\nPlease check and press enter to continue.:\n")

    ## Calculate F,E,R,t
    F, _ = cv.findFundamentalMat(image_point_pairs[0], image_point_pairs[1], cv.FM_RANSAC, 1, 0.99999)
    print(f"\nFundamental matrix:\n{F}")

    K1 = cameras.get_camera_params(0)["intrinsic_matrix"]
    K2 = cameras.get_camera_params(1)["intrinsic_matrix"]
    print(f"\nIntrinsic matrix K1:\n{K1}\n")
    print(f"\nIntrinsic matrix K2:\n{K2}\n")

    E = essential_from_fundamental(F, K1, K2) # OpenCV
    print(f'Essential matrix:\n{E}\n')

    Rs, ts = motion_from_essential(E)
    for i in range(4):
        print(f'\nRotations_matrices{i}:\n{Rs[i]}\ntranslations{i}:\n{ts[i]}')

    ## Choose the right combination of R,t
    correspondant_image_point_pairs = image_point_pairs.transpose(1,0,2).tolist()
    print(f"\ncorrespondant image point pairs:\n{correspondant_image_point_pairs}\n")
    positive_percentage = []
    for index in range(4):
        camera_poses = [
            {
                "R": np.eye(3),
                "t": np.array([0, 0, 0], dtype = np.float32)
            },
            {
                "R": Rs[index],
                "t": ts[index]
            }
        ]
        object_points = triangulate_points(correspondant_image_point_pairs, camera_poses)
        positive_count = (object_points[:,2] > 0).sum()
        positive_percentage.append(positive_count/len(object_points))
        print(f"index{index} positive percentage: {positive_count/len(object_points):.1%}")
    index = positive_percentage.index(max(positive_percentage))

    print(f"\nindex{index} is chosen\n")

    ## Bundle adjustment
    camera_poses = [
        {
            "R": np.eye(3),
            "t": np.array([0, 0, 0], dtype = np.float32)
        },
        {
            "R": Rs[index],
            "t": ts[index]
        }
    ]

    camera_poses = bundle_adjustment(correspondant_image_point_pairs, camera_poses)
    input("\nBundle adjustment complete, please check and press enter to continue.\n")

    ## Determine the scale of t
    actual_distance = 0.2
    num_scale_coef = 20
    scale_coef = []
    i = 0
    while i < num_scale_coef:
        input(f"\nPress enter to capture image{i} for the determination of t")
        frames = cameras.camera_read()
        image_point_0 = cameras.find_dot(frames[0])[1]
        image_point_1 = cameras.find_dot(frames[1])[1]
        if len(image_point_0) == 2 and len(image_point_1) == 2:
            image_points = [image_point_0, image_point_1]
            errors, object_points, _ = find_point_correspondance_and_object_points(image_points, camera_poses, frames)
            if len(object_points) == 2:
                mea_distance = np.sqrt(np.sum((object_points[0]-object_points[1])**2))
                scale_coef.append(actual_distance/mea_distance)
                print(f"scale_coef[{i}]:\n{scale_coef[i]}\n")
                i += 1
            else:
                print("Caculated object points <= 2\n")
                continue
        else:
            print("image points num wrong.\n")
            continue

    mean = np.mean(scale_coef)
    std = np.std(scale_coef)  
    scale_coef_average = np.mean([x for x in scale_coef if abs(x - mean) < 2 * std]) # 2σ rule
    print("Final scale coef:\n", scale_coef_average)

    camera_poses[1]["t"] = camera_poses[1]["t"] * scale_coef_average
    np.save("extrinsic/camera_poses.npy", camera_poses, allow_pickle=True)

    ## Establish world coordinate
    while True:
        input(f"Press enter to capture image for acquiring world coordinate\n")
        frames = cameras.camera_read()
        image_point_0 = cameras.find_dot(frames[0])[1]
        image_point_1 = cameras.find_dot(frames[1])[1]
        if len(image_point_0) == 3 and len(image_point_1) == 3:
            image_points = [image_point_0, image_point_1]
            errors, object_points, _ = find_point_correspondance_and_object_points(image_points, camera_poses, frames)

            print(type(frames[0]), frames[0].shape, frames[0].dtype)
            cv.imshow("image of camera0", frames[0])
            cv.waitKey(50)
            print(f"image points of camera0:\n{image_point_0}")
            if len(object_points) != 3:
                print("Calculated object points not enough.\n")
                continue
            while len(object_points) == 3:
                usr_input = input("Choose points to be the world coordinate's origin,x_axis,y_axis")
                numbers = usr_input.split()
                numbers = [int(num) for num in numbers]
                if all(0 <= num <= 2 for num in numbers) and len(numbers) == 3:
                    x_axis = object_points[numbers[1]] - object_points[numbers[0]]
                    y_axis = object_points[numbers[2]] - object_points[numbers[0]]
                    z_axis = np.cross(x_axis, y_axis)
                    y_axis = np.cross(z_axis, x_axis)

                    x_axis = x_axis / linalg.norm(x_axis)
                    y_axis = y_axis / linalg.norm(y_axis)
                    z_axis = z_axis / linalg.norm(z_axis)
                    
                    R = np.array([x_axis, y_axis, z_axis])
                    t = np.dot(R,-object_points[numbers[0]]).reshape(3,1)
                    camera_to_world_matrix = np.concatenate((np.concatenate((R,t),axis=1),np.array([[0,0,0,1]])),axis=0)
                    np.save("extrinsic/camera_to_world_matrix.npy", camera_to_world_matrix)
                    print(f"R:\n{R}\nt:\n{t}")
                    break
                else:
                    print("Wrong pattern!")
                    continue
            break
        else:
            print("\nimage points num wrong.\n")
            continue

def to_world_coordinate(object_points_camera: np.ndarray, camera_to_world_matrix) -> np.ndarray:
    if len(object_points_camera) == 0:
        return np.array([])
    else:
        Pc = np.concatenate((object_points_camera.transpose(1,0),np.ones((1,len(object_points_camera)))),axis=0)
        return np.dot(camera_to_world_matrix,Pc).transpose(1,0)[:, :-1]