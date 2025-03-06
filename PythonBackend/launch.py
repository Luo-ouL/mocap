from mocap.helpers import *
import matplotlib.pyplot as plt
from extrinsic_params_calculation import to_world_coordinate

np.set_printoptions(precision=8, suppress=True)
threold = 0.005

## Initialization
cameras = Cameras.instance()
camera_poses = np.load("extrinsic/camera_poses.npy", allow_pickle=True)
camera_to_world_matrix = np.load("extrinsic/camera_to_world_matrix.npy")
cameras.start_calculating_object_pose()

## Initialize frames visualizaion module
fourcc = cv.VideoWriter_fourcc(*'XVID') # using XVID coding method
out = cv.VideoWriter('Localization/frames_visualization_video.avi', fourcc, cameras.fps, (cameras.width*cameras.num_cameras, cameras.height))

## Initialize object points visualization module
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-0.5, 0.5)
ax.set_ylim(-0.5, 0.5)
ax.set_zlim(-0.5, 0.5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

## Get frames and image_points
while True:
# Output errors, object_points, frames
    frames = cameras.camera_read()
    image_points = []
    for cam_idx in range(cameras.num_cameras):
        image_points.append(cameras.find_dot(frames[cam_idx])[1])
    errors, object_points, _ = find_point_correspondance_and_object_points(image_points, camera_poses, frames)
    object_points = to_world_coordinate(object_points, camera_to_world_matrix)

    R, t, _ = calculate_object_pose(object_points, threold)
    print(f"translation:\n{t}\n")
    if len(R) == 1:
        print("euler angles:\n", np.array([]))
    else:
        r = Rotation.from_matrix(R.transpose(1,0))
        euler_angles = r.as_euler('zyx', degrees=True)
        print(f"euler angles:{euler_angles}\n")

    print(f'errors:\n{errors}\nobject_points:\n{object_points}\n')

# Frames visualization
    conbined_frame = np.concatenate(frames, axis=1)
    out.write(conbined_frame)
    cv.imshow('Correpondance', conbined_frame)

# Object points visualization
    # clear old graph
    ax.cla()
    # modify axis property and label
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(-0.5, 0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    if len(object_points) != 0:
        xs = [p[0] for p in object_points]
        ys = [p[1] for p in object_points]
        zs = [p[2] for p in object_points]
        ax.scatter(xs, ys, zs, c='red', marker='o', s=20)  # s controls the size of points

    # update graph
    plt.draw()

    if cv.waitKey(1) & 0xFF == ord('q'):
        break
