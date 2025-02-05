import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
import matplotlib.patches as mpatches
import numpy as np
from scipy.spatial.transform import Rotation as R

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

# Define a color for each category
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
    
    # Head frame: use EarL (1), EarR (2), and Snout (3)
    earL = get_kp(1)
    earR = get_kp(2)
    snout = get_kp(3)
    ear_mid = (earL + earR) / 2.0
    head_x = normalize(snout - ear_mid)
    temp_y = earR - earL
    head_y = normalize(temp_y - np.dot(temp_y, head_x) * head_x)
    head_z = normalize(np.cross(head_x, head_y))
    R_head = np.column_stack((head_x, head_y, head_z))
    
    # Body frame: use SpineF (4), SpineM (5), ShoulderL (12), ShoulderR (16)
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
times = []      # Timestamps from the DataFrame index
yaw_vals = []   # Relative yaw values (degrees)
pitch_vals = [] # Relative pitch values (degrees)
roll_vals = []  # Relative roll values (degrees)

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

# Unwrap the angles to remove discontinuities (convert to radians, unwrap, then back to degrees)
yaw_vals_unwrapped = np.degrees(np.unwrap(np.radians(yaw_vals)))
pitch_vals_unwrapped = np.degrees(np.unwrap(np.radians(pitch_vals)))
roll_vals_unwrapped = np.degrees(np.unwrap(np.radians(roll_vals)))

# -----------------------
# Set up the Figure with two subplots:
#  - Top: 3D animation of the skeleton and coordinate frames.
#  - Bottom: Line plot of Euler angles vs time.
# -----------------------
fig = plt.figure(figsize=(10, 10))
ax3d = fig.add_subplot(211, projection='3d')
ax_euler = fig.add_subplot(212)
plt.subplots_adjust(bottom=0.2, right=0.75, top=0.95, hspace=0.4)

# Global variable to track if the animation is running.
is_anim_running = False

# -----------------------
# Update function for animation (updates both subplots)
# -----------------------
def update(frame):
    # --- Update 3D subplot (ax3d) ---
    ax3d.clear()
    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Z')
    
    row = df.iloc[frame]
    timestamp = row.name
    
    # Plot keypoints with category colors and labels
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
    
    # Compute coordinate frames for the head and body:
    def get_kp(idx):
        return np.array([row[f'kp{idx}_x'], row[f'kp{idx}_y'], row[f'kp{idx}_z']])
    
    # --- Head frame ---
    # (For improved robustness, consider smoothing the keypoint positions over time.)
    earL = get_kp(1)
    earR = get_kp(2)
    snout = get_kp(3)
    ear_mid = (earL + earR) / 2.0
    head_x = normalize(snout - ear_mid)
    temp_y = earR - earL
    head_y = normalize(temp_y - np.dot(temp_y, head_x) * head_x)
    head_z = normalize(np.cross(head_x, head_y))
    R_head = np.column_stack((head_x, head_y, head_z))
    
    # --- Body frame ---
    # (Consider incorporating additional body points or smoothing if discontinuities persist.)
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
    yaw, pitch, roll = euler_angles
    text_str = f"Relative Euler Angles (deg):\nYaw: {yaw:.2f}, Pitch: {pitch:.2f}, Roll: {roll:.2f}"
    ax3d.text2D(0.05, 0.95, text_str, transform=ax3d.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(facecolor='w', alpha=0.5))
    
    # Draw coordinate frame arrows on ax3d
    arrow_length = 20  # Adjust as needed
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
    
    # Set equal aspect ratio for ax3d
    max_range = np.array([max(all_x)-min(all_x),
                          max(all_y)-min(all_y),
                          max(all_z)-min(all_z)]).max() / 2.0
    mid_x = (max(all_x)+min(all_x)) * 0.5
    mid_y = (max(all_y)+min(all_y)) * 0.5
    mid_z = (max(all_z)+min(all_z)) * 0.5
    ax3d.set_xlim(mid_x - max_range, mid_x + max_range)
    ax3d.set_ylim(mid_y - max_range, mid_y + max_range)
    ax3d.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # --- Update Euler angles subplot (ax_euler) ---
    ax_euler.clear()
    ax_euler.plot(times, yaw_vals_unwrapped, label='Yaw', color='blue')
    ax_euler.plot(times, pitch_vals_unwrapped, label='Pitch', color='green')
    ax_euler.plot(times, roll_vals_unwrapped, label='Roll', color='red')
    ax_euler.set_xlabel('Time')
    ax_euler.set_ylabel('Angle (deg)')
    ax_euler.set_title('Relative Euler Angles vs Time')
    ax_euler.legend(loc='upper right')
    # Draw a vertical dashed line at the current time
    ax_euler.axvline(x=timestamp, color='k', linestyle='--')
    
    # --- Update slider value if needed ---
    if int(frame_slider.val) != frame:
        frame_slider.set_val(frame)

# -----------------------
# Create the Slider widget (for frame control)
# -----------------------
slider_ax = fig.add_axes([0.15, 0.02, 0.55, 0.03])
frame_slider = Slider(slider_ax, 'Frame', 0, nframes-1, valinit=0, valfmt='%0.0f')

def slider_update(val):
    frame = int(frame_slider.val)
    update(frame)
    fig.canvas.draw_idle()
    # If the animation was running, resume it after a slider move.
    if is_anim_running:
        anim.event_source.start()

frame_slider.on_changed(slider_update)

# -----------------------
# Create Start and Stop buttons
# -----------------------
start_ax = fig.add_axes([0.75, 0.02, 0.1, 0.04])
stop_ax = fig.add_axes([0.75, 0.08, 0.1, 0.04])
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
# Create the animation
# -----------------------
anim = FuncAnimation(fig, update, frames=nframes, interval=100, repeat=True)
plt.show()
