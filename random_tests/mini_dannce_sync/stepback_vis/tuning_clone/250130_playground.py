import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Step 1: Generate synthetic keypoints data for the template and multiple frames
np.random.seed(42)

# Define a fake template frame with 4 keypoints (centroid at [0, 0, 0])
keypoints_template = np.array([
    [0.0, 0.0, 0.0],    # Centroid
    [1.0, 0.5, 0.0],    # Right ear
    [-1.0, 0.5, 0.0],   # Left ear
    [0.0, 1.5, 0.0]     # Snout
])

# Generate random transformations (for 5 frames)
frames_keypoints = []
for _ in range(5):
    # Apply a random rigid body transformation to the template
    rotation = R.from_euler('xyz', np.random.uniform(-20, 20, size=3), degrees=True)
    translation = np.random.uniform(-1, 1, size=3)
    transformed_points = rotation.apply(keypoints_template) + translation
    frames_keypoints.append(transformed_points)

# Convert to array: shape (frames, points, coordinates)
frames_keypoints = np.array(frames_keypoints)

# Step 2: Calculate the centroid for each frame
def calculate_centroid(points):
    return np.mean(points, axis=0)

centroids = np.array([calculate_centroid(frame) for frame in frames_keypoints])

# Step 3: Align each frame to the template using rigid body transformation
aligned_frames = []
for frame, centroid in zip(frames_keypoints, centroids):
    # Center the frame by subtracting its centroid
    centered_frame = frame - centroid

    # Find the optimal rotation to align the frame with the template
    rotation, _ = R.align_vectors(keypoints_template, centered_frame)
    aligned_frame = rotation.apply(centered_frame)

    aligned_frames.append(aligned_frame)

aligned_frames = np.array(aligned_frames)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot template points
ax.scatter(keypoints_template[:, 0], keypoints_template[:, 1], keypoints_template[:, 2], color='blue', label='Template')
for i, point in enumerate(keypoints_template):
    ax.text(point[0], point[1], point[2], f'T{i+1}', color='blue')

# Plot aligned frames with labels
colors = ['red', 'green', 'orange', 'purple', 'cyan']
for i, aligned_frame in enumerate(aligned_frames):
    ax.scatter(aligned_frame[:, 0] + i * 5, aligned_frame[:, 1], aligned_frame[:, 2], color=colors[i], label=f'Aligned Frame {i+1}')
    for j, point in enumerate(aligned_frame):
        ax.text(point[0] + i * 5, point[1], point[2], f'F{i+1}P{j+1}', color=colors[i])

# Set labels and legend
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.legend()
ax.set_title('Improved Rigid Body Transformation Visualization')
plt.show()



# import numpy as np
# from scipy.spatial.transform import Rotation as R
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # Step 1: Create fake head keypoints (template)
# keypoints_template = np.array([
#     [0.0, 0.0, 0.0],    # Centroid (origin)
#     [1.0, 0.0, 0.0],    # Right ear
#     [-1.0, 0.0, 0.0],   # Left ear
#     [0.0, 1.5, 0.0],    # Snout
#     [0.0, 0.0, -1.0]    # Neck point (back of head)
# ])

# # Step 2: Define a random rigid body transformation (rotation + translation)
# random_rotation = R.from_euler('xyz', [30, 45, 60], degrees=True)  # 30° pitch, 45° yaw, 60° roll
# translation_vector = np.array([2.0, -1.0, 3.0])  # Random translation

# # Apply the rigid body transformation
# keypoints_transformed = random_rotation.apply(keypoints_template) + translation_vector

# # Step 3: Visualization
# fig = plt.figure(figsize=(10, 6))
# ax = fig.add_subplot(111, projection='3d')

# # Plot template keypoints
# ax.scatter(keypoints_template[:, 0], keypoints_template[:, 1], keypoints_template[:, 2], color='blue', label='Template')
# for i, point in enumerate(keypoints_template):
#     ax.text(point[0], point[1], point[2], f'P{i+1}', color='blue')

# # Plot transformed keypoints
# ax.scatter(keypoints_transformed[:, 0], keypoints_transformed[:, 1], keypoints_transformed[:, 2], color='red', label='Transformed')
# for i, point in enumerate(keypoints_transformed):
#     ax.text(point[0], point[1], point[2], f'P{i+1}', color='red')

# # Connect corresponding points with lines
# for i in range(len(keypoints_template)):
#     ax.plot([keypoints_template[i, 0], keypoints_transformed[i, 0]],
#             [keypoints_template[i, 1], keypoints_transformed[i, 1]],
#             [keypoints_template[i, 2], keypoints_transformed[i, 2]], color='gray', linestyle='--')

# # Set plot labels
# ax.set_xlabel('X-axis')
# ax.set_ylabel('Y-axis')
# ax.set_zlabel('Z-axis')
# ax.legend()
# ax.set_title('Rigid Body Transformation of a Fake Head')
# plt.show()
