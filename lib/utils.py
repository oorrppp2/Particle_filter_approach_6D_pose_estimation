import numpy as np
import math
from scipy.spatial import distance, cKDTree
import cv2
import json

# epsilon for testing whether a number is close to zero
_EPS = np.finfo(float).eps * 4.0

# The number of colors has to be more than the number of target objects for visualization.
color = [[255, 255, 255], [0, 255, 0], [255, 0, 0], [0, 0, 255], [255, 255, 0],
        [255, 0, 255], [0, 255, 255], [128, 0, 0], [0, 128, 0], [0, 0, 128],
        [251, 194, 44], [240, 20, 134], [160, 103, 173], [70, 163, 210], [140, 227, 61],
        [128, 128, 0], [128, 0, 128], [0, 128, 128], [64, 0, 0], [0, 64, 0], [0, 0, 64]]

def str2bool(v):
    if isinstance(v, bool): 
        return v 
    if v.lower() in ('yes', 'true', 't', 'y', '1'): 
        return True 
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): 
        return False 
    else: 
        raise argparse.ArgumentTypeError('Boolean value expected.')

def draw_object(itemid, K, my_r, my_t, img, model_points, corner):
    img_width, img_length, _ = img.shape
    pred = np.dot(model_points, my_r.T) + my_t

    pred_box = np.dot(corner, my_r.T) + my_t
    transposed_pred_box = pred_box.T
    pred_box = transposed_pred_box/transposed_pred_box[2,:]
    pred_box_pixel = K @ pred_box
    pred_box_pixel = pred_box_pixel.astype(np.int64)

    cv2.line(img, (pred_box_pixel[0, 0], pred_box_pixel[1, 0]),
        (pred_box_pixel[0, 1], pred_box_pixel[1, 1]), (0,0,255), 2, lineType=cv2.LINE_AA)
    cv2.line(img, (pred_box_pixel[0, 1], pred_box_pixel[1, 1]),
        (pred_box_pixel[0, 2], pred_box_pixel[1, 2]), (0,0,255), 2, lineType=cv2.LINE_AA)
    cv2.line(img, (pred_box_pixel[0, 2], pred_box_pixel[1, 2]),
        (pred_box_pixel[0, 3], pred_box_pixel[1, 3]), (0,0,255), 2, lineType=cv2.LINE_AA)
    cv2.line(img, (pred_box_pixel[0, 3], pred_box_pixel[1, 3]),
        (pred_box_pixel[0, 0], pred_box_pixel[1, 0]), (0,0,255), 2, lineType=cv2.LINE_AA)

    cv2.line(img, (pred_box_pixel[0, 4], pred_box_pixel[1, 4]),
        (pred_box_pixel[0, 5], pred_box_pixel[1, 5]), (0,0,255), 2, lineType=cv2.LINE_AA)
    cv2.line(img, (pred_box_pixel[0, 5], pred_box_pixel[1, 5]),
        (pred_box_pixel[0, 6], pred_box_pixel[1, 6]), (0,0,255), 2, lineType=cv2.LINE_AA)
    cv2.line(img, (pred_box_pixel[0, 6], pred_box_pixel[1, 6]),
        (pred_box_pixel[0, 7], pred_box_pixel[1, 7]), (0,0,255), 2, lineType=cv2.LINE_AA)
    cv2.line(img, (pred_box_pixel[0, 7], pred_box_pixel[1, 7]),
        (pred_box_pixel[0, 4], pred_box_pixel[1, 4]), (0,0,255), 2, lineType=cv2.LINE_AA)

    cv2.line(img, (pred_box_pixel[0, 0], pred_box_pixel[1, 0]),
        (pred_box_pixel[0, 4], pred_box_pixel[1, 4]), (0,0,255), 2, lineType=cv2.LINE_AA)
    cv2.line(img, (pred_box_pixel[0, 1], pred_box_pixel[1, 1]),
        (pred_box_pixel[0, 5], pred_box_pixel[1, 5]), (0,0,255), 2, lineType=cv2.LINE_AA)
    cv2.line(img, (pred_box_pixel[0, 2], pred_box_pixel[1, 2]),
        (pred_box_pixel[0, 6], pred_box_pixel[1, 6]), (0,0,255), 2, lineType=cv2.LINE_AA)
    cv2.line(img, (pred_box_pixel[0, 3], pred_box_pixel[1, 3]),
        (pred_box_pixel[0, 7], pred_box_pixel[1, 7]), (0,0,255), 2, lineType=cv2.LINE_AA)

    transposed_pred = pred.T
    pred = transposed_pred/transposed_pred[2,:]
    pred_pixel = K @ pred
    pred_pixel = pred_pixel.astype(np.int64)

    _, cols = pred_pixel.shape
    del_list = []
    for i in range(cols):
        if pred_pixel[0,i] >= img_length or pred_pixel[1,i] >= img_width or \
                pred_pixel[0, i] < 0 or pred_pixel[1, i] < 0 :
            del_list.append(i)
    pred_pixel = np.delete(pred_pixel, del_list, axis=1)

    # Large dots
    img[pred_pixel[1,:]+1, pred_pixel[0,:]] = color[int(itemid-1)]
    img[pred_pixel[1,:]-1, pred_pixel[0,:]] = color[int(itemid-1)]
    img[pred_pixel[1,:], pred_pixel[0,:]+1] = color[int(itemid-1)]
    img[pred_pixel[1,:], pred_pixel[0,:]-1] = color[int(itemid-1)]
    img[pred_pixel[1,:], pred_pixel[0,:]] = color[int(itemid-1)]
    img[pred_pixel[1,:]+1, pred_pixel[0,:]+1] = color[int(itemid-1)]
    img[pred_pixel[1,:]-1, pred_pixel[0,:]-1] = color[int(itemid-1)]
    img[pred_pixel[1,:]+1, pred_pixel[0,:]-1] = color[int(itemid-1)]
    img[pred_pixel[1,:]-1, pred_pixel[0,:]+1] = color[int(itemid-1)]

    # Small dots
    # img[pred_pixel[1,:], pred_pixel[0,:]] = color[int(itemid-1)]
    return img


def quaternion_matrix(quaternion):
  """Return homogeneous rotation matrix from quaternion.

  >>> M = quaternion_matrix([0.99810947, 0.06146124, 0, 0])
  >>> numpy.allclose(M, rotation_matrix(0.123, [1, 0, 0]))
  True
  >>> M = quaternion_matrix([1, 0, 0, 0])
  >>> numpy.allclose(M, numpy.identity(4))
  True
  >>> M = quaternion_matrix([0, 1, 0, 0])
  >>> numpy.allclose(M, numpy.diag([1, -1, -1, 1]))
  True

  """
  q = np.array(quaternion, dtype=np.float64, copy=True)
  n = np.dot(q, q)
  if n < _EPS:
    return np.identity(4)
  q *= math.sqrt(2.0 / n)
  q = np.outer(q, q)
  return np.array([
    [1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0], 0.0],
    [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0], 0.0],
    [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2], 0.0],
    [0.0, 0.0, 0.0, 1.0]])


def calc_pts_diameter(pts):
  """Calculates the diameter of a set of 3D points (i.e. the maximum distance
  between any two points in the set). Faster but requires more memory than
  calc_pts_diameter.

  :param pts: nx3 ndarray with 3D points.
  :return: The calculated diameter.
  """
  dists = distance.cdist(pts, pts, 'euclidean')
  diameter = np.max(dists)
  return diameter

def transform_pts_Rt(pts, R, t):
  """Applies a rigid transformation to 3D points.

  :param pts: nx3 ndarray with 3D points.
  :param R: 3x3 ndarray with a rotation matrix.
  :param t: 3x1 ndarray with a translation vector.
  :return: nx3 ndarray with transformed 3D points.
  """
  assert (pts.shape[1] == 3)
  pts_t = R.dot(pts.T) + t.reshape((3, 1))
  return pts_t.T

def adi(R_est, t_est, R_gt, t_gt, pts):
  """Average Distance of Model Points for objects with indistinguishable views
  - by Hinterstoisser et al. (ACCV'12).

  :param R_est: 3x3 ndarray with the estimated rotation matrix.
  :param t_est: 3x1 ndarray with the estimated translation vector.
  :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
  :param t_gt: 3x1 ndarray with the ground-truth translation vector.
  :param pts: nx3 ndarray with 3D model points.
  :return: The calculated error.
  """
  pts_est = transform_pts_Rt(pts, R_est, t_est)
  pts_gt = transform_pts_Rt(pts, R_gt, t_gt)

  # Calculate distances to the nearest neighbors from vertices in the
  # ground-truth pose to vertices in the estimated pose.
  nn_index = spatial.cKDTree(pts_est)
  nn_dists, _ = nn_index.query(pts_gt, k=1)

  e = nn_dists.mean()
  return e

def add(R_est, t_est, R_gt, t_gt, pts, diameter):
  """Average Distance of Model Points for objects with no indistinguishable
  views - by Hinterstoisser et al. (ACCV'12).

  :param R_est: 3x3 ndarray with the estimated rotation matrix.
  :param t_est: 3x1 ndarray with the estimated translation vector.
  :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
  :param t_gt: 3x1 ndarray with the ground-truth translation vector.
  :param pts: nx3 ndarray with 3D model points.
  :return: The calculated error.
  """
  pts_est = transform_pts_Rt(pts, R_est, t_est)
  pts_gt = transform_pts_Rt(pts, R_gt, t_gt)
  e = np.linalg.norm(pts_est - pts_gt, axis=1).mean()
  return e, e < diameter

def adi(R_est, t_est, R_gt, t_gt, pts, diameter):
  """Average Distance of Model Points for objects with indistinguishable views
  - by Hinterstoisser et al. (ACCV'12).

  :param R_est: 3x3 ndarray with the estimated rotation matrix.
  :param t_est: 3x1 ndarray with the estimated translation vector.
  :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
  :param t_gt: 3x1 ndarray with the ground-truth translation vector.
  :param pts: nx3 ndarray with 3D model points.
  :return: The calculated error.
  """
  pts_est = transform_pts_Rt(pts, R_est, t_est)
  pts_gt = transform_pts_Rt(pts, R_gt, t_gt)

  # Calculate distances to the nearest neighbors from vertices in the
  # ground-truth pose to vertices in the estimated pose.
  nn_index = cKDTree(pts_est)
  nn_dists, _ = nn_index.query(pts_gt, k=1)

  e = nn_dists.mean()
  return e, e < diameter

def compute_pose_metrics(rec, max_auc_dist = 0.1, max_pose_dist = 0.02):
    # TODO : this should be in utils.py
    '''
    Follows plot_accuracy_keyframe.m from YCB_Video_toolbox
        @rec - np.array - add-s values in sorted order
        @prec - accuracy number
    '''
    rec_mean = np.mean(rec)
    rec_less = np.where(rec < max_pose_dist)[0]
    rec_less_perc = rec_less.shape[0]/rec.shape[0] * 100.0

    rec[rec > max_auc_dist] = np.inf
    rec = np.sort(rec)
    prec = np.arange(0, rec.shape[0], 1)/rec.shape[0]
    # Remove first 0 and add 1 at the end (denotes 100 percent of poses)
    prec = np.array(prec[1:].tolist() + [1])

    index = np.isfinite(rec)
    # Actual pose error
    rec = rec[index]
    # Percentage of poses with that error
    prec = prec[index]

    # Append end point values
    mrec = np.array([0] + rec.tolist() + [0.1])
    mpre = np.array([0] + prec.tolist() + [prec[-1]])

    # Indexes where value is not equal to previous value
    args = np.where(mrec[:-1] != mrec[1:])[0]
    args_prev = args
    args = args + 1

    # Calculate area under the curve
    ap = np.sum((mrec[args] - mrec[args_prev]) * mpre[args]) * 10

    return {
        "auc" : ap * 100.0,
        "pose_error_less_perc" : rec_less_perc,
        "mean_pose_error" : rec_mean,
        "pose_count" : rec.shape[0],
        # "mrec" : mrec,
        # "mpre" : mpre
    }

def load_json(path, keys_to_int=False):
  """Loads content of a JSON file.

  :param path: Path to the JSON file.
  :return: Content of the loaded JSON file.
  """
  # Keys to integers.
  def convert_keys_to_int(x):
    return {int(k) if k.lstrip('-').isdigit() else k: v for k, v in x.items()}

  with open(path, 'r') as f:
    if keys_to_int:
      content = json.load(f, object_hook=lambda x: convert_keys_to_int(x))
    else:
      content = json.load(f)

  return content

def load_scene_gt(path):
  """Loads content of a JSON file with ground-truth annotations.

  See docs/bop_datasets_format.md for details.

  :param path: Path to the JSON file.
  :return: Dictionary with the loaded content.
  """
  scene_gt = load_json(path, keys_to_int=True)

  for im_id, im_gt in scene_gt.items():
    for gt in im_gt:
      if 'cam_R_m2c' in gt.keys():
        gt['cam_R_m2c'] = np.array(gt['cam_R_m2c'], np.float).reshape((3, 3))
      if 'cam_t_m2c' in gt.keys():
        gt['cam_t_m2c'] = np.array(gt['cam_t_m2c'], np.float).reshape((3, 1))
  return scene_gt
