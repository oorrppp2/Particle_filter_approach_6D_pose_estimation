from dipy.core.geometry import cart2sphere, sphere2cart
import math
import numpy as np

def sphere2euler(theta, phi):
    theta = math.pi/2 - theta
    return theta, phi

def sphere_fibonacci_grid_points_with_sym_metric (ng, half_whole):

  if (half_whole == 1):
    rnd = 1.
    samples  = ng
    randomize = False
    if randomize:
        rnd = random.random() * samples

    points = []
    offset = 2./samples
    increment = math.pi * (3. - math.sqrt(5.));

    for i in range(samples):
        y = ((i * offset) - 1) + (offset / 2);
        r = math.sqrt(1 - pow(y,2))

        phi = ((i + rnd) % samples) * increment

        x = math.cos(phi) * r
        z = math.sin(phi) * r

        points.append([x,y,z])

    return np.array(points)  
  else: # half_whole == 0
    rnd = 1.
    samples  = ng
    randomize = False
    if randomize:
        rnd = random.random() * samples

    points = []
    offset = 2./samples
    increment = math.pi * (3. - math.sqrt(5.));

    for i in range(round(samples/2)):
        y = ((i * offset) - 1) + (offset / 2);
        r = math.sqrt(1 - pow(y,2))

        phi = ((i + rnd) % samples) * increment

        x = math.cos(phi) * r
        z = math.sin(phi) * r

        points.append([x,y,z])

    return np.array(points)

def get_rotation_samples(obj_shape, num_samples):
    all_rots = []
    
    viewpoints_xyz = sphere_fibonacci_grid_points_with_sym_metric(num_samples,1)
    for viewpoint in viewpoints_xyz:
        r, theta, phi = cart2sphere(viewpoint[0], viewpoint[1], viewpoint[2])
        theta, phi = sphere2euler(theta, phi)
        if obj_shape == "non-symmetric":
            step_size = math.pi/4
            for yaw_temp in np.arange(0,math.pi, step_size):
                xyz_rotation_angles = [-phi, yaw_temp, theta]
                all_rots.append(xyz_rotation_angles)
        elif obj_shape == "symmetric":
            xyz_rotation_angles = [-phi, theta, 0]
            all_rots.append(xyz_rotation_angles)
        elif obj_shape == "half-symmetric":
            step_size = math.pi/2
            for yaw_temp in np.arange(0,math.pi, step_size):
                xyz_rotation_angles = [-phi, yaw_temp, theta]
                all_rots.append(xyz_rotation_angles)
    return all_rots



def get_rotation_samples_perch(label, num_samples):
    all_rots = []
    name_sym_dict = {
        # Discribe the symmetric feature of the object: 
        # First Item is for the sphere symmetric. the second is for the yaw
        # Second for changing raw. Here may need to rewrite the render or transition matrix!!!!!!!
        # First (half: 0, whole: 1) Second (0:0, 1:0-pi, 2:0-2pi)
        "002_master_chef_can":      [0,0], #half_0
        "003_cracker_box":          [0,0], #half_0-pi #0,0 is fine with gicp
        "004_sugar_box":            [0,3], #half_0-pi
        "005_tomato_soup_can":      [0,0], #half_0
        "006_mustard_bottle":       [0,0], #whole_0-pi
        "007_tuna_fish_can":        [0,0], #half_0
        "008_pudding_box":          [0,1], #half_0-pi
        "009_gelatin_box":          [0,0], #half_0-pi
        "010_potted_meat_can":      [0,0], #half_0-pi
        "011_banana":               [1,0], #whole_0-2pi #from psc
        "019_pitcher_base":         [0,0], #whole_0-2pi
        "021_bleach_cleanser":      [0,0], #whole_0-2pi, 55 and 
        "024_bowl":                 [1,0], #whole_0
        "025_mug":                  [0,1], #whole_0-2pi
        "035_power_drill" :         [0,7], #whole_0-2pi
        "036_wood_block":           [0,0], #half_0-pi
        "037_scissors":             [0,2], #whole_0-2pi
        "040_large_marker" :        [1,0], #whole_0
        "052_extra_large_clamp":    [0,7],  #whole_0-pi
        "051_large_clamp":          [0,7],
        "061_foam_brick":           [0,0], #half_0-pi
        "water_bottle":             [0,0], #half_0-pi
        "coca_cola":                [0,0], #half_0-pi


        # # LM-O
        # "Ape":          [0,7], #half_0-pi
        # "Can":          [0,7], #whole_0-2pi
        # "Cat" :         [0,7], #whole_0
        # "Driller":      [0,7],  #whole_0-pi
        # "Duck":         [0,7],
        # "Eggbox":       [0,7], #half_0-pi
        # "Glue":         [0,7],
        # "Holepuncher":  [0,7], #half_0-pi

    }
    
    viewpoints_xyz = sphere_fibonacci_grid_points_with_sym_metric(num_samples,name_sym_dict[label][0])
    for viewpoint in viewpoints_xyz:
        r, theta, phi = cart2sphere(viewpoint[0], viewpoint[1], viewpoint[2])
        theta, phi = sphere2euler(theta, phi)
        if name_sym_dict[label][1] == 0:
            xyz_rotation_angles = [-phi, theta, 0]
            all_rots.append(xyz_rotation_angles)
        elif name_sym_dict[label][1] == 1:
            step_size = math.pi/2
            for yaw_temp in np.arange(0,math.pi, step_size):
                xyz_rotation_angles = [-phi, yaw_temp, theta]
                all_rots.append(xyz_rotation_angles)
        elif name_sym_dict[label][1] == 2:
            step_size = math.pi/4
            for yaw_temp in np.arange(0,math.pi, step_size):
                xyz_rotation_angles = [-phi, yaw_temp, theta]
                all_rots.append(xyz_rotation_angles)
        elif name_sym_dict[label][1] == 3:
            xyz_rotation_angles = [-phi, 0, theta]
            all_rots.append(xyz_rotation_angles)
            xyz_rotation_angles = [-phi, 2*math.pi/3, theta]
            all_rots.append(xyz_rotation_angles)
        elif name_sym_dict[label][1] == 4:
            # For upright sugar box
            xyz_rotation_angles = [-phi, math.pi+theta, 0]
            all_rots.append(xyz_rotation_angles)
        elif name_sym_dict[label][1] == 5:
            xyz_rotation_angles = [phi, theta, math.pi]
            all_rots.append(xyz_rotation_angles)
        elif name_sym_dict[label][1] == 6:
            # This causes sampling of inplane along z
            xyz_rotation_angles = [-phi, 0, theta]
            all_rots.append(xyz_rotation_angles)
            xyz_rotation_angles = [-phi, math.pi/3, theta]
            all_rots.append(xyz_rotation_angles)
            xyz_rotation_angles = [-phi, 2*math.pi/3, theta]
            all_rots.append(xyz_rotation_angles)
        elif name_sym_dict[label][1] == 7:
            step_size = math.pi/2
            for yaw_temp in np.arange(0, 2*math.pi, step_size):
                xyz_rotation_angles = [-phi, yaw_temp, theta]
                all_rots.append(xyz_rotation_angles)
        elif name_sym_dict[label][1] == 8:
            step_size = math.pi/3
            for yaw_temp in np.arange(0, math.pi, step_size):
                xyz_rotation_angles = [yaw_temp, -phi, theta]
                all_rots.append(xyz_rotation_angles)

    return all_rots
