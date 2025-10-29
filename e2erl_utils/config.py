import numpy as np

initial_xd = 1.5
xd_base = 15
xd_scale = 5


# actions
steering_action = np.pi / 8
speed_action_low = initial_xd - (2 * xd_scale)
speed_action_high = initial_xd + (2 * xd_scale)

# observations
steering_observation = np.pi / 6
hitch_angle_observation = np.pi / 2
speed_observation = 20
cross_track_angle_observation = np.pi / 4
cross_track_distance_observation = 100
curvature_observation = 2
yaw_rate_observation = 1.5

# Error Scale Constants
error_scale = 1
error_theta_scale = 1
error_speed_scale = 1
yaw_rate_error_scale = 1
curvature_scale = 2

# Misc Constants
max_yaw_rate = np.deg2rad(15)

tesla_model_s_vehicle_params = {
        'm': 1500,  # Mass of the vehicle (kg)
        'Iz': 3000,  # Moment of inertia (kg.m^2)
        'Cf': 80000,  # Cornering stiffness of front tires (N/rad)
        'Cr': 80000,  # Cornering stiffness of rear tires (N/rad)
        'lf': 1.2,  # Distance from CG to front axle (m)
        'lr': 1.6,  # Distance from CG to rear axle (m)
        'Cd': 0.208,  # Drag Coefficient
        'A': 2.4,  # Drag Cross-Section area (m^2)
        'dt': 0.1  # Sampling time (s)
}

record_data = False
use_piecewise_curve = False
visualize = True

environment_render_mode = 1 #1 is render.py, #2 is render_observation.py needed for the CNN based DDPG models

grid_res_m = 0.10                     # meters per cell
lane_centerline_half_width_m = 5.0    # half of the lane width (e.g., ~3.5 m lane)
lane_shoulder_m = 0.50                # extra margin on each side
lane_sample_ds_m = 0.25

# --- BEV camera ---
bev_anchor = "tractor_rear_axle"   # "tractor_cg" | "tractor_rear_axle" | "trailer_axle"
bev_offset_x_m = -4.0       # in tractor body frame (+x forward); negative = bias behind tractor
bev_offset_y_m = 0.0        # lateral offset (rarely needed)
bev_forward_up = True       # if True, rotate so tractor forward points UP on screen
use_bev_render = True       # toggle BEV vs global view
bev_zoom_scale = 3.0