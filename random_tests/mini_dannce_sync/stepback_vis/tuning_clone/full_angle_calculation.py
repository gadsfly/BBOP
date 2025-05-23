import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
import matplotlib.patches as mpatches
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.gridspec as gridspec

# -----------------------
# Load the DataFrame
# -----------------------
hdf5_file_path = "/data/big_rim/rsync_dcc_sum/Oct3V1/2024_10_25/20241002PMCr2_17_05/MIR_Aligned/aligned_predictions_with_ca_and_dF_F.h5"
df = pd.read_hdf(hdf5_file_path, key='df')

# -----------------------
# Define keypoint labels in order (1-indexed)
# -----------------------
keypoint_labels = [
    'EarL',       # 1
    'EarR',       # 2
    'Snout',      # 3
    'SpineF',     # 4
    'SpineM',     # 5
    'Tail(base)', # 6
    'Tail(mid)',  # 7
    'Tail(end)',  # 8
    'ForepawL',   # 9
    'WristL',     # 10
    'ElbowL',     # 11
    'ShoulderL',  # 12
    'ForepawR',   # 13
    'WristR',     # 14
    'ElbowR',     # 15
    'ShoulderR',  # 16
    'HindpawL',   # 17
    'AnkleL',     # 18
    'KneeL',      # 19
    'HindpawR',   # 20
    'AnkleR',     # 21
    'KneeR'       # 22
]

# -----------------------
# Define anatomical categories and assign keypoints to each
# -----------------------
categories = {
    'Ears': ['EarL', 'EarR'],
    'Snout': ['Snout'],
    'Spine': ['SpineF', 'SpineM'],
    'Tail': ['Tail(base)', 'Tail(mid)', 'Tail(end)'],
    'Limbs': ['ForepawL', 'WristL', 'ElbowL', 'ShoulderL',
              'ForepawR', 'WristR', 'ElbowR', 'ShoulderR'],
    'Paws': ['HindpawL', 'AnkleL', 'KneeL', 'HindpawR', 'AnkleR', 'KneeR']
}

# Colors for each category
category_colors = {
    'Ears': 'red',
    'Snout': 'brown',
    'Spine': 'green',
    'Tail': 'blue',
    'Limbs': 'purple',
    'Paws': 'orange'
}

# -----------------------
# Helper function: normalize a vector
# -----------------------
def normalize(v):
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        return v  # avoid division by zero
    return v / norm

# -----------------------
# Function to compute Euler angles from a DataFrame row
# -----------------------
def compute_euler_angles_from_row(row):
    def get_kp(idx):
        return np.array([row[f'kp{idx}_x'], row[f'kp{idx}_y'], row[f'kp{idx}_z']])
    
    # Head frame: using EarL (1), EarR (2), and Snout (3)
    earL = get_kp(1)
    earR = get_kp(2)
    snout = get_kp(3)
    ear_mid = (earL + earR) / 2.0
    head_x = normalize(snout - ear_mid)
    temp_y = earR - earL
    head_y = normalize(temp_y - np.dot(temp_y, head_x) * head_x)
    head_z = normalize(np.cross(head_x, head_y))
    R_head = np.column_stack((head_x, head_y, head_z))
    
    # Body frame: using SpineF (4), SpineM (5), ShoulderL (12), ShoulderR (16)
    spineF = get_kp(4)
    spineM = get_kp(5)
    shoulderL = get_kp(12)
    shoulderR = get_kp(16)
    body_origin = spineM.copy()
    body_x = normalize(spineF - spineM)
    shoulder_vec = shoulderR - shoulderL
    body_y = normalize(shoulder_vec - np.dot(shoulder_vec, body_x) * body_x)
    body_z = normalize(np.cross(body_x, body_y))
    R_body = np.column_stack((body_x, body_y, body_z))
    
    R_rel = np.dot(R_body.T, R_head)
    euler_angles = R.from_matrix(R_rel).as_euler('xyz', degrees=True)
    return euler_angles  # [yaw, pitch, roll]

# -----------------------
# Precompute Euler angles for all frames
# -----------------------
nframes = len(df)
times = []      # timestamps (from DataFrame index)
yaw_vals = []   # relative yaw values (degrees)
pitch_vals = [] # relative pitch values (degrees)
roll_vals = []  # relative roll values (degrees)

for i in range(nframes):
    row = df.iloc[i]
    times.append(row.name)
    euler = compute_euler_angles_from_row(row)
    yaw_vals.append(euler[0])
    pitch_vals.append(euler[1])
    roll_vals.append(euler[2])

yaw_vals = np.array(yaw_vals)
pitch_vals = np.array(pitch_vals)
roll_vals = np.array(roll_vals)

# Unwrap the angles to remove discontinuities
yaw_vals_unwrapped = np.degrees(np.unwrap(np.radians(yaw_vals)))
pitch_vals_unwrapped = np.degrees(np.unwrap(np.radians(pitch_vals)))
roll_vals_unwrapped = np.degrees(np.unwrap(np.radians(roll_vals)))

# -----------------------
# Compute Back Pitch for Each Frame
# -----------------------
# We extend the line from spineF (keypoint 4) to spineM (keypoint 5) until it
# intersects with the horizontal plane z = 0. Then, we compute two vectors:
#   1. vector_spine: from the intersection point I to spineM.
#   2. vector_horizontal: the horizontal projection (in the xy-plane) of that same vector.
# The angle between these vectors is the back pitch.
back_pitch_vals = []
for i in range(nframes):
    row = df.iloc[i]
    spineF = np.array([row['kp4_x'], row['kp4_y'], row['kp4_z']])
    spineM = np.array([row['kp5_x'], row['kp5_y'], row['kp5_z']])
    
    # Parametric line: L(t) = spineM + t*(spineF - spineM)
    # Find t such that z = 0: spineM_z + t*(spineF_z - spineM_z) = 0
    delta_z = spineF[2] - spineM[2]
    if np.abs(delta_z) < 1e-8:
        I = spineM.copy()  # already nearly horizontal
    else:
        t = -spineM[2] / delta_z
        I = spineM + t * (spineF - spineM)
    
    vector_spine = spineM - I  
    vector_horizontal = np.array([vector_spine[0], vector_spine[1], 0])
    
    norm_spine = np.linalg.norm(vector_spine)
    norm_horizontal = np.linalg.norm(vector_horizontal)
    
    if norm_spine < 1e-8 or norm_horizontal < 1e-8:
        angle_deg = 0.0
    else:
        cos_angle = np.clip(np.dot(vector_spine, vector_horizontal) / (norm_spine * norm_horizontal), -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(cos_angle))
    
    # Optionally, assign a sign based on whether spineM is above or below the horizontal.
    if vector_spine[2] < 0:
        angle_deg = -angle_deg
        
    back_pitch_vals.append(angle_deg)

# -----------------------
# Set up the Figure with a Tiled Layout (2 rows x 2 columns):
#
#   Top Left: 3D skeleton animation.
#   Bottom Left: 3D back pitch visualization.
#   Top Right: Euler angles vs time.
#   Bottom Right: Back pitch vs time.
# -----------------------
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1])
ax3d = fig.add_subplot(gs[0, 0], projection='3d')     # Top left: 3D skeleton
ax_back3d = fig.add_subplot(gs[1, 0], projection='3d')  # Bottom left: 3D back pitch
ax_euler = fig.add_subplot(gs[0, 1])                    # Top right: Euler angles vs time
ax_back = fig.add_subplot(gs[1, 1])                     # Bottom right: Back pitch vs time

plt.subplots_adjust(bottom=0.15, right=0.9, top=0.95, wspace=0.3, hspace=0.3)

# Global variable to track if the animation is running.
is_anim_running = False

# -----------------------
# Update function for the animation (updates all subplots)
# -----------------------
def update(frame):
    # -------------------
    # Update 3D Skeleton Animation (ax3d)
    # -------------------
    ax3d.clear()
    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Z')
    
    row = df.iloc[frame]
    timestamp = row.name
    
    # Plot keypoints with category colors and labels.
    all_x, all_y, all_z = [], [], []
    for cat, labels in categories.items():
        color = category_colors[cat]
        for label in labels:
            idx = keypoint_labels.index(label) + 1
            x = row[f'kp{idx}_x']
            y = row[f'kp{idx}_y']
            z = row[f'kp{idx}_z']
            all_x.append(x)
            all_y.append(y)
            all_z.append(z)
            ax3d.scatter(x, y, z, c=color, marker='o', s=30)
            ax3d.text(x, y, z, label, color=color, fontsize=8)
    
    ax3d.set_title(f'Timestamp: {timestamp}')
    handles = [mpatches.Patch(color=category_colors[cat], label=cat) for cat in categories]
    ax3d.legend(handles=handles, loc='upper left')
    
    def get_kp(idx):
        return np.array([row[f'kp{idx}_x'], row[f'kp{idx}_y'], row[f'kp{idx}_z']])
    
    # Compute head coordinate frame.
    earL = get_kp(1)
    earR = get_kp(2)
    snout = get_kp(3)
    ear_mid = (earL + earR) / 2.0
    head_x = normalize(snout - ear_mid)
    temp_y = earR - earL
    head_y = normalize(temp_y - np.dot(temp_y, head_x) * head_x)
    head_z = normalize(np.cross(head_x, head_y))
    # Compute body coordinate frame.
    spineF = get_kp(4)
    spineM = get_kp(5)
    shoulderL = get_kp(12)
    shoulderR = get_kp(16)
    body_origin = spineM.copy()
    body_x = normalize(spineF - spineM)
    shoulder_vec = shoulderR - shoulderL
    body_y = normalize(shoulder_vec - np.dot(shoulder_vec, body_x) * body_x)
    body_z = normalize(np.cross(body_x, body_y))
    
    # Relative rotation and Euler angles.
    R_rel = np.dot(np.column_stack((body_x, body_y, body_z)).T, 
                   np.column_stack((head_x, head_y, head_z)))
    euler_angles = R.from_matrix(R_rel).as_euler('xyz', degrees=True)
    yaw, pitch, roll = euler_angles
    text_str = f"Relative Euler Angles (deg):\nYaw: {yaw:.2f}, Pitch: {pitch:.2f}, Roll: {roll:.2f}"
    ax3d.text2D(0.05, 0.95, text_str, transform=ax3d.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(facecolor='w', alpha=0.5))
    
    arrow_length = 20  # Adjust as needed.
    ax3d.quiver(body_origin[0], body_origin[1], body_origin[2],
                body_x[0], body_x[1], body_x[2],
                length=arrow_length, color='blue', normalize=True)
    ax3d.quiver(body_origin[0], body_origin[1], body_origin[2],
                body_y[0], body_y[1], body_y[2],
                length=arrow_length, color='green', normalize=True)
    ax3d.quiver(body_origin[0], body_origin[1], body_origin[2],
                body_z[0], body_z[1], body_z[2],
                length=arrow_length, color='purple', normalize=True)
    ax3d.quiver(ear_mid[0], ear_mid[1], ear_mid[2],
                head_x[0], head_x[1], head_x[2],
                length=arrow_length, color='cyan', normalize=True)
    ax3d.quiver(ear_mid[0], ear_mid[1], ear_mid[2],
                head_y[0], head_y[1], head_y[2],
                length=arrow_length, color='magenta', normalize=True)
    ax3d.quiver(ear_mid[0], ear_mid[1], ear_mid[2],
                head_z[0], head_z[1], head_z[2],
                length=arrow_length, color='orange', normalize=True)
    
    # Set equal aspect based on keypoint extents.
    max_range = np.array([max(all_x)-min(all_x),
                          max(all_y)-min(all_y),
                          max(all_z)-min(all_z)]).max() / 2.0
    mid_x = (max(all_x)+min(all_x)) * 0.5
    mid_y = (max(all_y)+min(all_y)) * 0.5
    mid_z = (max(all_z)+min(all_z)) * 0.5
    ax3d.set_xlim(mid_x - max_range, mid_x + max_range)
    ax3d.set_ylim(mid_y - max_range, mid_y + max_range)
    ax3d.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # -------------------
    # Update 3D Back Pitch Visualization (ax_back3d)
    # -------------------
    ax_back3d.clear()
    ax_back3d.set_xlabel('X')
    ax_back3d.set_ylabel('Y')
    ax_back3d.set_zlabel('Z')
    ax_back3d.set_title('Back Pitch Visualization')
    
    # For the back pitch, we use spineF (keypoint 4) and spineM (keypoint 5)
    spineF = get_kp(4)
    spineM = get_kp(5)
    # Compute the intersection (I) of the line (spineM to spineF) with the plane z = 0.
    delta_z = spineF[2] - spineM[2]
    if np.abs(delta_z) < 1e-8:
        I = spineM.copy()
    else:
        t = -spineM[2] / delta_z
        I = spineM + t * (spineF - spineM)
    
    # Compute vectors:
    vector_spine = spineM - I  
    vector_horizontal = np.array([vector_spine[0], vector_spine[1], 0])
    
    # Plot the horizontal plane z = 0.
    margin = 20
    x_min = spineM[0] - margin
    x_max = spineM[0] + margin
    y_min = spineM[1] - margin
    y_max = spineM[1] + margin
    X, Y = np.meshgrid(np.linspace(x_min, x_max, 2),
                       np.linspace(y_min, y_max, 2))
    Z = np.zeros_like(X)
    ax_back3d.plot_surface(X, Y, Z, alpha=0.3, color='gray')
    
    # Plot spine points and the intersection.
    ax_back3d.scatter(spineF[0], spineF[1], spineF[2], color='red', s=50, label='SpineF')
    ax_back3d.scatter(spineM[0], spineM[1], spineM[2], color='green', s=50, label='SpineM')
    ax_back3d.scatter(I[0], I[1], I[2], color='blue', s=50, label='Intersection (I)')
    
    # Draw vectors from I:
    ax_back3d.quiver(I[0], I[1], I[2],
                     vector_spine[0], vector_spine[1], vector_spine[2],
                     color='magenta', arrow_length_ratio=0.1, linewidth=2, label='Spine Vector')
    ax_back3d.quiver(I[0], I[1], I[2],
                     vector_horizontal[0], vector_horizontal[1], vector_horizontal[2],
                     color='orange', arrow_length_ratio=0.1, linewidth=2, label='Horizontal Projection')
    
    # Optionally, plot a line joining spineF and spineM.
    ax_back3d.plot([spineF[0], spineM[0]], [spineF[1], spineM[1]], [spineF[2], spineM[2]],
                   linestyle='--', color='black', label='Spine Line')
    
    # Set equal aspect limits for ax_back3d.
    all_pts = np.array([spineF, spineM, I])
    xlims = [all_pts[:, 0].min()-margin, all_pts[:, 0].max()+margin]
    ylims = [all_pts[:, 1].min()-margin, all_pts[:, 1].max()+margin]
    zlims = [min(all_pts[:, 2].min(), 0)-margin, all_pts[:, 2].max()+margin]
    ax_back3d.set_xlim(xlims)
    ax_back3d.set_ylim(ylims)
    ax_back3d.set_zlim(zlims)
    
    # Display the back pitch angle as computed earlier.
    bp_angle = back_pitch_vals[frame]
    ax_back3d.text2D(0.05, 0.95, f"Back Pitch: {bp_angle:.2f}°", transform=ax_back3d.transAxes,
                     fontsize=10, bbox=dict(facecolor='w', alpha=0.5))
    
    # -------------------
    # Update Euler Angles vs Time (ax_euler)
    # -------------------
    ax_euler.clear()
    ax_euler.plot(times, yaw_vals_unwrapped, label='Yaw', color='blue')
    ax_euler.plot(times, pitch_vals_unwrapped, label='Pitch', color='green')
    ax_euler.plot(times, roll_vals_unwrapped, label='Roll', color='red')
    ax_euler.set_xlabel('Time')
    ax_euler.set_ylabel('Angle (deg)')
    ax_euler.set_title('Relative Euler Angles vs Time')
    ax_euler.legend(loc='upper right')
    ax_euler.axvline(x=timestamp, color='k', linestyle='--')
    
    # -------------------
    # Update Back Pitch vs Time (ax_back)
    # -------------------
    ax_back.clear()
    ax_back.plot(times, back_pitch_vals, label='Back Pitch', color='magenta')
    ax_back.set_xlabel('Time')
    ax_back.set_ylabel('Back Pitch (deg)')
    ax_back.set_title('Back Pitch vs Time')
    ax_back.legend(loc='upper right')
    ax_back.axvline(x=timestamp, color='k', linestyle='--')
    
    # Update slider position if needed.
    if int(frame_slider.val) != frame:
        frame_slider.set_val(frame)

# -----------------------
# Create the Slider widget and Start/Stop Buttons
# -----------------------
slider_ax = fig.add_axes([0.15, 0.03, 0.55, 0.03])
frame_slider = Slider(slider_ax, 'Frame', 0, nframes-1, valinit=0, valfmt='%0.0f')

def slider_update(val):
    frame = int(frame_slider.val)
    update(frame)
    fig.canvas.draw_idle()
    if is_anim_running:
        anim.event_source.start()

frame_slider.on_changed(slider_update)

start_ax = fig.add_axes([0.75, 0.03, 0.1, 0.04])
stop_ax = fig.add_axes([0.75, 0.09, 0.1, 0.04])
start_button = Button(start_ax, 'Start')
stop_button = Button(stop_ax, 'Stop')

def start(event):
    global is_anim_running
    is_anim_running = True
    anim.event_source.start()

def stop(event):
    global is_anim_running
    is_anim_running = False
    anim.event_source.stop()

start_button.on_clicked(start)
stop_button.on_clicked(stop)

# -----------------------
# Create and start the animation
# -----------------------
anim = FuncAnimation(fig, update, frames=nframes, interval=100, repeat=True)
plt.show()
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
import matplotlib.patches as mpatches
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.gridspec as gridspec

# -----------------------
# Load the DataFrame
# -----------------------
hdf5_file_path = "/data/big_rim/rsync_dcc_sum/Oct3V1/2024_10_25/20241002PMCr2_17_05/MIR_Aligned/aligned_predictions_with_ca_and_dF_F.h5"

# "C:/Users/shiny/Desktop/ShinySw/code/aligned_predictions_with_ca_and_dF_F.h5"
df = pd.read_hdf(hdf5_file_path, key='df')

# -----------------------
# Define keypoint labels in order (1-indexed)
# -----------------------
keypoint_labels = [
    'EarL',       # 1
    'EarR',       # 2
    'Snout',      # 3
    'SpineF',     # 4
    'SpineM',     # 5
    'Tail(base)', # 6
    'Tail(mid)',  # 7
    'Tail(end)',  # 8
    'ForepawL',   # 9
    'WristL',     # 10
    'ElbowL',     # 11
    'ShoulderL',  # 12
    'ForepawR',   # 13
    'WristR',     # 14
    'ElbowR',     # 15
    'ShoulderR',  # 16
    'HindpawL',   # 17
    'AnkleL',     # 18
    'KneeL',      # 19
    'HindpawR',   # 20
    'AnkleR',     # 21
    'KneeR'       # 22
]

# -----------------------
# Define anatomical categories and assign keypoints to each
# -----------------------
categories = {
    'Ears': ['EarL', 'EarR'],
    'Snout': ['Snout'],
    'Spine': ['SpineF', 'SpineM'],
    'Tail': ['Tail(base)', 'Tail(mid)', 'Tail(end)'],
    'Limbs': ['ForepawL', 'WristL', 'ElbowL', 'ShoulderL',
              'ForepawR', 'WristR', 'ElbowR', 'ShoulderR'],
    'Paws': ['HindpawL', 'AnkleL', 'KneeL', 'HindpawR', 'AnkleR', 'KneeR']
}

# Colors for each category
category_colors = {
    'Ears': 'red',
    'Snout': 'brown',
    'Spine': 'green',
    'Tail': 'blue',
    'Limbs': 'purple',
    'Paws': 'orange'
}

# -----------------------
# Helper function: normalize a vector
# -----------------------
def normalize(v):
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        return v  # avoid division by zero
    return v / norm

# -----------------------
# Function to compute Euler angles from a DataFrame row
# -----------------------
def compute_euler_angles_from_row(row):
    def get_kp(idx):
        return np.array([row[f'kp{idx}_x'], row[f'kp{idx}_y'], row[f'kp{idx}_z']])
    
    # Head frame: using EarL (1), EarR (2), and Snout (3)
    earL = get_kp(1)
    earR = get_kp(2)
    snout = get_kp(3)
    ear_mid = (earL + earR) / 2.0
    head_x = normalize(snout - ear_mid)
    temp_y = earR - earL
    head_y = normalize(temp_y - np.dot(temp_y, head_x) * head_x)
    head_z = normalize(np.cross(head_x, head_y))
    R_head = np.column_stack((head_x, head_y, head_z))
    
    # Body frame: using SpineF (4), SpineM (5), ShoulderL (12), ShoulderR (16)
    spineF = get_kp(4)
    spineM = get_kp(5)
    shoulderL = get_kp(12)
    shoulderR = get_kp(16)
    body_origin = spineM.copy()
    body_x = normalize(spineF - spineM)
    shoulder_vec = shoulderR - shoulderL
    body_y = normalize(shoulder_vec - np.dot(shoulder_vec, body_x) * body_x)
    body_z = normalize(np.cross(body_x, body_y))
    R_body = np.column_stack((body_x, body_y, body_z))
    
    R_rel = np.dot(R_body.T, R_head)
    euler_angles = R.from_matrix(R_rel).as_euler('xyz', degrees=True)
    return euler_angles  # [yaw, pitch, roll]

# -----------------------
# Precompute Euler angles for all frames
# -----------------------
nframes = len(df)
times = []      # timestamps (from DataFrame index)
yaw_vals = []   # relative yaw values (degrees)
pitch_vals = [] # relative pitch values (degrees)
roll_vals = []  # relative roll values (degrees)

for i in range(nframes):
    row = df.iloc[i]
    times.append(row.name)
    euler = compute_euler_angles_from_row(row)
    yaw_vals.append(euler[0])
    pitch_vals.append(euler[1])
    roll_vals.append(euler[2])

yaw_vals = np.array(yaw_vals)
pitch_vals = np.array(pitch_vals)
roll_vals = np.array(roll_vals)

# Unwrap the angles to remove discontinuities
yaw_vals_unwrapped = np.degrees(np.unwrap(np.radians(yaw_vals)))
pitch_vals_unwrapped = np.degrees(np.unwrap(np.radians(pitch_vals)))
roll_vals_unwrapped = np.degrees(np.unwrap(np.radians(roll_vals)))

# -----------------------
# Compute Back Pitch for Each Frame
# -----------------------
# We extend the line from spineF (keypoint 4) to spineM (keypoint 5) until it
# intersects with the horizontal plane z = 0. Then, we compute two vectors:
#   1. vector_spine: from the intersection point I to spineM.
#   2. vector_horizontal: the horizontal projection (in the xy-plane) of that same vector.
# The angle between these vectors is the back pitch.
back_pitch_vals = []
for i in range(nframes):
    row = df.iloc[i]
    spineF = np.array([row['kp4_x'], row['kp4_y'], row['kp4_z']])
    spineM = np.array([row['kp5_x'], row['kp5_y'], row['kp5_z']])
    
    # Parametric line: L(t) = spineM + t*(spineF - spineM)
    # Find t such that z = 0: spineM_z + t*(spineF_z - spineM_z) = 0
    delta_z = spineF[2] - spineM[2]
    if np.abs(delta_z) < 1e-8:
        I = spineM.copy()  # already nearly horizontal
    else:
        t = -spineM[2] / delta_z
        I = spineM + t * (spineF - spineM)
    
    vector_spine = spineM - I  
    vector_horizontal = np.array([vector_spine[0], vector_spine[1], 0])
    
    norm_spine = np.linalg.norm(vector_spine)
    norm_horizontal = np.linalg.norm(vector_horizontal)
    
    if norm_spine < 1e-8 or norm_horizontal < 1e-8:
        angle_deg = 0.0
    else:
        cos_angle = np.clip(np.dot(vector_spine, vector_horizontal) / (norm_spine * norm_horizontal), -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(cos_angle))
    
    # Optionally, assign a sign based on whether spineM is above or below the horizontal.
    if vector_spine[2] < 0:
        angle_deg = -angle_deg
        
    back_pitch_vals.append(angle_deg)

# -----------------------
# Set up the Figure with a Tiled Layout (2 rows x 2 columns):
#
#   Top Left: 3D skeleton animation.
#   Bottom Left: 3D back pitch visualization.
#   Top Right: Euler angles vs time.
#   Bottom Right: Back pitch vs time.
# -----------------------
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1])
ax3d = fig.add_subplot(gs[0, 0], projection='3d')     # Top left: 3D skeleton
ax_back3d = fig.add_subplot(gs[1, 0], projection='3d')  # Bottom left: 3D back pitch
ax_euler = fig.add_subplot(gs[0, 1])                    # Top right: Euler angles vs time
ax_back = fig.add_subplot(gs[1, 1])                     # Bottom right: Back pitch vs time

plt.subplots_adjust(bottom=0.15, right=0.9, top=0.95, wspace=0.3, hspace=0.3)

# Global variable to track if the animation is running.
is_anim_running = False

# -----------------------
# Update function for the animation (updates all subplots)
# -----------------------
def update(frame):
    # -------------------
    # Update 3D Skeleton Animation (ax3d)
    # -------------------
    ax3d.clear()
    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Z')
    
    row = df.iloc[frame]
    timestamp = row.name
    
    # Plot keypoints with category colors and labels.
    all_x, all_y, all_z = [], [], []
    for cat, labels in categories.items():
        color = category_colors[cat]
        for label in labels:
            idx = keypoint_labels.index(label) + 1
            x = row[f'kp{idx}_x']
            y = row[f'kp{idx}_y']
            z = row[f'kp{idx}_z']
            all_x.append(x)
            all_y.append(y)
            all_z.append(z)
            ax3d.scatter(x, y, z, c=color, marker='o', s=30)
            ax3d.text(x, y, z, label, color=color, fontsize=8)
    
    ax3d.set_title(f'Timestamp: {timestamp}')
    handles = [mpatches.Patch(color=category_colors[cat], label=cat) for cat in categories]
    ax3d.legend(handles=handles, loc='upper left')
    
    def get_kp(idx):
        return np.array([row[f'kp{idx}_x'], row[f'kp{idx}_y'], row[f'kp{idx}_z']])
    
    # Compute head coordinate frame.
    earL = get_kp(1)
    earR = get_kp(2)
    snout = get_kp(3)
    ear_mid = (earL + earR) / 2.0
    head_x = normalize(snout - ear_mid)
    temp_y = earR - earL
    head_y = normalize(temp_y - np.dot(temp_y, head_x) * head_x)
    head_z = normalize(np.cross(head_x, head_y))
    # Compute body coordinate frame.
    spineF = get_kp(4)
    spineM = get_kp(5)
    shoulderL = get_kp(12)
    shoulderR = get_kp(16)
    body_origin = spineM.copy()
    body_x = normalize(spineF - spineM)
    shoulder_vec = shoulderR - shoulderL
    body_y = normalize(shoulder_vec - np.dot(shoulder_vec, body_x) * body_x)
    body_z = normalize(np.cross(body_x, body_y))
    
    # Relative rotation and Euler angles.
    R_rel = np.dot(np.column_stack((body_x, body_y, body_z)).T, 
                   np.column_stack((head_x, head_y, head_z)))
    euler_angles = R.from_matrix(R_rel).as_euler('xyz', degrees=True)
    yaw, pitch, roll = euler_angles
    text_str = f"Relative Euler Angles (deg):\nYaw: {yaw:.2f}, Pitch: {pitch:.2f}, Roll: {roll:.2f}"
    ax3d.text2D(0.05, 0.95, text_str, transform=ax3d.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(facecolor='w', alpha=0.5))
    
    arrow_length = 20  # Adjust as needed.
    ax3d.quiver(body_origin[0], body_origin[1], body_origin[2],
                body_x[0], body_x[1], body_x[2],
                length=arrow_length, color='blue', normalize=True)
    ax3d.quiver(body_origin[0], body_origin[1], body_origin[2],
                body_y[0], body_y[1], body_y[2],
                length=arrow_length, color='green', normalize=True)
    ax3d.quiver(body_origin[0], body_origin[1], body_origin[2],
                body_z[0], body_z[1], body_z[2],
                length=arrow_length, color='purple', normalize=True)
    ax3d.quiver(ear_mid[0], ear_mid[1], ear_mid[2],
                head_x[0], head_x[1], head_x[2],
                length=arrow_length, color='cyan', normalize=True)
    ax3d.quiver(ear_mid[0], ear_mid[1], ear_mid[2],
                head_y[0], head_y[1], head_y[2],
                length=arrow_length, color='magenta', normalize=True)
    ax3d.quiver(ear_mid[0], ear_mid[1], ear_mid[2],
                head_z[0], head_z[1], head_z[2],
                length=arrow_length, color='orange', normalize=True)
    
    # Set equal aspect based on keypoint extents.
    max_range = np.array([max(all_x)-min(all_x),
                          max(all_y)-min(all_y),
                          max(all_z)-min(all_z)]).max() / 2.0
    mid_x = (max(all_x)+min(all_x)) * 0.5
    mid_y = (max(all_y)+min(all_y)) * 0.5
    mid_z = (max(all_z)+min(all_z)) * 0.5
    ax3d.set_xlim(mid_x - max_range, mid_x + max_range)
    ax3d.set_ylim(mid_y - max_range, mid_y + max_range)
    ax3d.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # -------------------
    # Update 3D Back Pitch Visualization (ax_back3d)
    # -------------------
    ax_back3d.clear()
    ax_back3d.set_xlabel('X')
    ax_back3d.set_ylabel('Y')
    ax_back3d.set_zlabel('Z')
    ax_back3d.set_title('Back Pitch Visualization')
    
    # For the back pitch, we use spineF (keypoint 4) and spineM (keypoint 5)
    spineF = get_kp(4)
    spineM = get_kp(5)
    # Compute the intersection (I) of the line (spineM to spineF) with the plane z = 0.
    delta_z = spineF[2] - spineM[2]
    if np.abs(delta_z) < 1e-8:
        I = spineM.copy()
    else:
        t = -spineM[2] / delta_z
        I = spineM + t * (spineF - spineM)
    
    # Compute vectors:
    vector_spine = spineM - I  
    vector_horizontal = np.array([vector_spine[0], vector_spine[1], 0])
    
    # Plot the horizontal plane z = 0.
    margin = 20
    x_min = spineM[0] - margin
    x_max = spineM[0] + margin
    y_min = spineM[1] - margin
    y_max = spineM[1] + margin
    X, Y = np.meshgrid(np.linspace(x_min, x_max, 2),
                       np.linspace(y_min, y_max, 2))
    Z = np.zeros_like(X)
    ax_back3d.plot_surface(X, Y, Z, alpha=0.3, color='gray')
    
    # Plot spine points and the intersection.
    ax_back3d.scatter(spineF[0], spineF[1], spineF[2], color='red', s=50, label='SpineF')
    ax_back3d.scatter(spineM[0], spineM[1], spineM[2], color='green', s=50, label='SpineM')
    ax_back3d.scatter(I[0], I[1], I[2], color='blue', s=50, label='Intersection (I)')
    
    # Draw vectors from I:
    ax_back3d.quiver(I[0], I[1], I[2],
                     vector_spine[0], vector_spine[1], vector_spine[2],
                     color='magenta', arrow_length_ratio=0.1, linewidth=2, label='Spine Vector')
    ax_back3d.quiver(I[0], I[1], I[2],
                     vector_horizontal[0], vector_horizontal[1], vector_horizontal[2],
                     color='orange', arrow_length_ratio=0.1, linewidth=2, label='Horizontal Projection')
    
    # Optionally, plot a line joining spineF and spineM.
    ax_back3d.plot([spineF[0], spineM[0]], [spineF[1], spineM[1]], [spineF[2], spineM[2]],
                   linestyle='--', color='black', label='Spine Line')
    
    # Set equal aspect limits for ax_back3d.
    all_pts = np.array([spineF, spineM, I])
    xlims = [all_pts[:, 0].min()-margin, all_pts[:, 0].max()+margin]
    ylims = [all_pts[:, 1].min()-margin, all_pts[:, 1].max()+margin]
    zlims = [min(all_pts[:, 2].min(), 0)-margin, all_pts[:, 2].max()+margin]
    ax_back3d.set_xlim(xlims)
    ax_back3d.set_ylim(ylims)
    ax_back3d.set_zlim(zlims)
    
    # Display the back pitch angle as computed earlier.
    bp_angle = back_pitch_vals[frame]
    ax_back3d.text2D(0.05, 0.95, f"Back Pitch: {bp_angle:.2f}°", transform=ax_back3d.transAxes,
                     fontsize=10, bbox=dict(facecolor='w', alpha=0.5))
    
    # -------------------
    # Update Euler Angles vs Time (ax_euler)
    # -------------------
    ax_euler.clear()
    ax_euler.plot(times, yaw_vals_unwrapped, label='Yaw', color='blue')
    ax_euler.plot(times, pitch_vals_unwrapped, label='Pitch', color='green')
    ax_euler.plot(times, roll_vals_unwrapped, label='Roll', color='red')
    ax_euler.set_xlabel('Time')
    ax_euler.set_ylabel('Angle (deg)')
    ax_euler.set_title('Relative Euler Angles vs Time')
    ax_euler.legend(loc='upper right')
    ax_euler.axvline(x=timestamp, color='k', linestyle='--')
    
    # -------------------
    # Update Back Pitch vs Time (ax_back)
    # -------------------
    ax_back.clear()
    ax_back.plot(times, back_pitch_vals, label='Back Pitch', color='magenta')
    ax_back.set_xlabel('Time')
    ax_back.set_ylabel('Back Pitch (deg)')
    ax_back.set_title('Back Pitch vs Time')
    ax_back.legend(loc='upper right')
    ax_back.axvline(x=timestamp, color='k', linestyle='--')
    
    # Update slider position if needed.
    if int(frame_slider.val) != frame:
        frame_slider.set_val(frame)

# -----------------------
# Create the Slider widget and Start/Stop Buttons
# -----------------------
slider_ax = fig.add_axes([0.15, 0.03, 0.55, 0.03])
frame_slider = Slider(slider_ax, 'Frame', 0, nframes-1, valinit=0, valfmt='%0.0f')

def slider_update(val):
    frame = int(frame_slider.val)
    update(frame)
    fig.canvas.draw_idle()
    if is_anim_running:
        anim.event_source.start()

frame_slider.on_changed(slider_update)

start_ax = fig.add_axes([0.75, 0.03, 0.1, 0.04])
stop_ax = fig.add_axes([0.75, 0.09, 0.1, 0.04])
start_button = Button(start_ax, 'Start')
stop_button = Button(stop_ax, 'Stop')

def start(event):
    global is_anim_running
    is_anim_running = True
    anim.event_source.start()

def stop(event):
    global is_anim_running
    is_anim_running = False
    anim.event_source.stop()

start_button.on_clicked(start)
stop_button.on_clicked(stop)

# -----------------------
# Create and start the animation
# -----------------------
anim = FuncAnimation(fig, update, frames=nframes, interval=100, repeat=True)
plt.show()
