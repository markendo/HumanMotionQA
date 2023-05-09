import sys, os

import json
import os.path as osp

import glob
import os
import argparse
import time

import numpy as np
import torch

import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

from body_model.body_model import BodyModel
from body_model.utils import SMPL_JOINTS, KEYPT_VERTS
from utils.torch import copy2cpu as c2c
from utils.transforms import batch_rodrigues, compute_world2aligned_mat, compute_world2aligned_joints_mat, axisangle2matrots, convert_to_rotmat

from tqdm import tqdm
#
# Processing options
#

OUT_FPS = 30
SAVE_KEYPT_VERTS = False # save vertex locations of certain keypoints
SAVE_HAND_POSE = False # save joint angles for the hand
SAVE_VELOCITIES = False # save all parameter velocities available
SAVE_ALIGN_ROT = False # save rot mats that go from world root orient to aligned root orient
DISCARD_TERRAIN_SEQUENCES = False # throw away sequences where the person steps onto objects (determined by a heuristic)

# if sequence is longer than this, splits into sequences of this size to avoid running out of memory
# ~ 4000 for 12 GB GPU, ~2000 for 8 GB
SPLIT_FRAME_LIMIT = 2000

NUM_BETAS = 16 # size of SMPL shape parameter to use

# for determining floor height
FLOOR_VEL_THRESH = 0.005
FLOOR_HEIGHT_OFFSET = 0.01
# for determining contacts
CONTACT_VEL_THRESH = 0.005 #0.015
CONTACT_TOE_HEIGHT_THRESH = 0.04
CONTACT_ANKLE_HEIGHT_THRESH = 0.08
# for determining terrain interaction
TERRAIN_HEIGHT_THRESH = 0.04 # if static toe is above this height
ROOT_HEIGHT_THRESH = 0.04 # if maximum "static" root height is more than this + root_floor_height
CLUSTER_SIZE_THRESH = 0.25 # if cluster has more than this faction of fps (30 for 120 fps)

DISCARD_SHORTER_THAN_FRAMES = 2 # 1 frame will cause an error

# Used for MotionClip baseline
SMPLH_JOINT_NAMES = ['pelvis','left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee', 'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot', 'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_index1', 'left_index2', 'left_index3', 'left_middle1', 'left_middle2', 'left_middle3', 'left_pinky1', 'left_pinky2', 'left_pinky3', 'left_ring1', 'left_ring2', 'left_ring3', 'left_thumb1', 'left_thumb2', 'left_thumb3', 'right_index1', 'right_index2', 'right_index3', 'right_middle1', 'right_middle2', 'right_middle3', 'right_pinky1', 'right_pinky2', 'right_pinky3', 'right_ring1', 'right_ring2', 'right_ring3', 'right_thumb1', 'right_thumb2', 'right_thumb3', 'nose', 'right_eye', 'left_eye', 'right_ear', 'left_ear', 'left_big_toe', 'left_small_toe', 'left_heel', 'right_big_toe', 'right_small_toe', 'right_heel', 'left_thumb', 'left_index', 'left_middle', 'left_ring', 'left_pinky', 'right_thumb', 'right_index', 'right_middle', 'right_ring', 'right_pinky',]
def get_joints_to_use():
    joints_to_use = np.array([
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        11, 12, 13, 14, 15, 16, 17, 18, 19,
        20, 21, 22, 37
    ])  # 23 joints + global_orient # 21 base joints + left_index1(22) + right_index1 (37)
    return np.arange(0, len(SMPLH_JOINT_NAMES) * 3).reshape((-1, 3))[joints_to_use].reshape(-1)
action2motion_joints = [8, 1, 2, 3, 4, 5, 6, 7, 0, 9, 10, 11, 12, 13, 14, 21, 24, 38]
ROT_CONVENTION_TO_ROT_NUMBER = {
'legacy': 23,
'no_hands': 21,
'full_hands': 51,
'mitten_hands': 33,
}
def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    return matrix[..., :2, :].clone().reshape(*matrix.size()[:-2], 6)
def axis_angle_to_matrix(axis_angle):
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))
def quaternion_to_matrix(quaternions):
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))
def axis_angle_to_quaternion(axis_angle):
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = 0.5 * angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions

#
# Processing
#

def get_body_model_sequence(smplh_path, gender, num_frames,
                  pose_body, pose_hand, betas, root_orient, trans):
    gender = str(gender)
    bm_path = os.path.join(smplh_path, gender + '/model.npz')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bm = BodyModel(bm_path=bm_path, num_betas=NUM_BETAS, batch_size=num_frames).to(device)

    pose_body = torch.Tensor(pose_body).to(device)
    pose_hand = torch.Tensor(pose_hand).to(device)
    betas = torch.Tensor(np.repeat(betas[:NUM_BETAS][np.newaxis], num_frames, axis=0)).to(device)
    root_orient = torch.Tensor(root_orient).to(device)
    trans = torch.Tensor(trans).to(device)
    body = bm(pose_body=pose_body, pose_hand=pose_hand, betas=betas, root_orient=root_orient, trans=trans)
    return body

def determine_floor_height_and_contacts(body_joint_seq, fps):
    '''
    Input: body_joint_seq N x 21 x 3 numpy array
    Contacts are N x 4 where N is number of frames and each row is left heel/toe, right heel/toe
    '''
    num_frames = body_joint_seq.shape[0]

    # compute toe velocities
    root_seq = body_joint_seq[:, SMPL_JOINTS['hips'], :]
    left_toe_seq = body_joint_seq[:, SMPL_JOINTS['leftToeBase'], :]
    right_toe_seq = body_joint_seq[:, SMPL_JOINTS['rightToeBase'], :]
    left_toe_vel = np.linalg.norm(left_toe_seq[1:] - left_toe_seq[:-1], axis=1)
    left_toe_vel = np.append(left_toe_vel, left_toe_vel[-1])
    right_toe_vel = np.linalg.norm(right_toe_seq[1:] - right_toe_seq[:-1], axis=1)
    right_toe_vel = np.append(right_toe_vel, right_toe_vel[-1])

    # now foot heights (z is up)
    left_toe_heights = left_toe_seq[:, 2]
    right_toe_heights = right_toe_seq[:, 2]
    root_heights = root_seq[:, 2]

    # filter out heights when velocity is greater than some threshold (not in contact)
    all_inds = np.arange(left_toe_heights.shape[0])
    left_static_foot_heights = left_toe_heights[left_toe_vel < FLOOR_VEL_THRESH]
    left_static_inds = all_inds[left_toe_vel < FLOOR_VEL_THRESH]
    right_static_foot_heights = right_toe_heights[right_toe_vel < FLOOR_VEL_THRESH]
    right_static_inds = all_inds[right_toe_vel < FLOOR_VEL_THRESH]

    all_static_foot_heights = np.append(left_static_foot_heights, right_static_foot_heights)
    all_static_inds = np.append(left_static_inds, right_static_inds)

    discard_seq = False
    if all_static_foot_heights.shape[0] > 0:
        cluster_heights = []
        cluster_root_heights = []
        cluster_sizes = []
        # cluster foot heights and find one with smallest median
        clustering = DBSCAN(eps=0.005, min_samples=3).fit(all_static_foot_heights.reshape(-1, 1))
        all_labels = np.unique(clustering.labels_)
        min_median = min_root_median = float('inf')
        for cur_label in all_labels:
            cur_clust = all_static_foot_heights[clustering.labels_ == cur_label]
            cur_clust_inds = np.unique(all_static_inds[clustering.labels_ == cur_label]) # inds in the original sequence that correspond to this cluster
            # get median foot height and use this as height
            cur_median = np.median(cur_clust)
            cluster_heights.append(cur_median)
            cluster_sizes.append(cur_clust.shape[0])

            # get root information
            cur_root_clust = root_heights[cur_clust_inds]
            cur_root_median = np.median(cur_root_clust)
            cluster_root_heights.append(cur_root_median)

            # update min info
            if cur_median < min_median:
                min_median = cur_median
                min_root_median = cur_root_median

        floor_height = min_median 
        offset_floor_height = floor_height - FLOOR_HEIGHT_OFFSET # toe joint is actually inside foot mesh a bit

        if DISCARD_TERRAIN_SEQUENCES:
            for cluster_root_height, cluster_height, cluster_size in zip (cluster_root_heights, cluster_heights, cluster_sizes):
                root_above_thresh = cluster_root_height > (min_root_median + ROOT_HEIGHT_THRESH)
                toe_above_thresh = cluster_height > (min_median + TERRAIN_HEIGHT_THRESH)
                cluster_size_above_thresh = cluster_size > int(CLUSTER_SIZE_THRESH*fps)
                if root_above_thresh and toe_above_thresh and cluster_size_above_thresh:
                    discard_seq = True
                    print('DISCARDING sequence based on terrain interaction!')
                    break
    else:
        floor_height = offset_floor_height = 0.0

    # now find contacts (feet are below certain velocity and within certain range of floor)
    # compute heel velocities
    left_heel_seq = body_joint_seq[:, SMPL_JOINTS['leftFoot'], :]
    right_heel_seq = body_joint_seq[:, SMPL_JOINTS['rightFoot'], :]
    left_heel_vel = np.linalg.norm(left_heel_seq[1:] - left_heel_seq[:-1], axis=1)
    left_heel_vel = np.append(left_heel_vel, left_heel_vel[-1])
    right_heel_vel = np.linalg.norm(right_heel_seq[1:] - right_heel_seq[:-1], axis=1)
    right_heel_vel = np.append(right_heel_vel, right_heel_vel[-1])

    left_heel_contact = left_heel_vel < CONTACT_VEL_THRESH
    right_heel_contact = right_heel_vel < CONTACT_VEL_THRESH
    left_toe_contact = left_toe_vel < CONTACT_VEL_THRESH
    right_toe_contact = right_toe_vel < CONTACT_VEL_THRESH

    # compute heel heights
    left_heel_heights = left_heel_seq[:, 2] - floor_height
    right_heel_heights = right_heel_seq[:, 2] - floor_height
    left_toe_heights =  left_toe_heights - floor_height
    right_toe_heights =  right_toe_heights - floor_height

    left_heel_contact = np.logical_and(left_heel_contact, left_heel_heights < CONTACT_ANKLE_HEIGHT_THRESH)
    right_heel_contact = np.logical_and(right_heel_contact, right_heel_heights < CONTACT_ANKLE_HEIGHT_THRESH)
    left_toe_contact = np.logical_and(left_toe_contact, left_toe_heights < CONTACT_TOE_HEIGHT_THRESH)
    right_toe_contact = np.logical_and(right_toe_contact, right_toe_heights < CONTACT_TOE_HEIGHT_THRESH)

    contacts = np.zeros((num_frames, len(SMPL_JOINTS)))
    contacts[:,SMPL_JOINTS['leftFoot']] = left_heel_contact
    contacts[:,SMPL_JOINTS['leftToeBase']] = left_toe_contact
    contacts[:,SMPL_JOINTS['rightFoot']] = right_heel_contact
    contacts[:,SMPL_JOINTS['rightToeBase']] = right_toe_contact

    # hand contacts
    left_hand_contact = detect_joint_contact(body_joint_seq, 'leftHand', floor_height, CONTACT_VEL_THRESH, CONTACT_ANKLE_HEIGHT_THRESH)
    right_hand_contact = detect_joint_contact(body_joint_seq, 'rightHand', floor_height, CONTACT_VEL_THRESH, CONTACT_ANKLE_HEIGHT_THRESH)
    contacts[:,SMPL_JOINTS['leftHand']] = left_hand_contact
    contacts[:,SMPL_JOINTS['rightHand']] = right_hand_contact

    # knee contacts
    left_knee_contact = detect_joint_contact(body_joint_seq, 'leftLeg', floor_height, CONTACT_VEL_THRESH, CONTACT_ANKLE_HEIGHT_THRESH)
    right_knee_contact = detect_joint_contact(body_joint_seq, 'rightLeg', floor_height, CONTACT_VEL_THRESH, CONTACT_ANKLE_HEIGHT_THRESH)
    contacts[:,SMPL_JOINTS['leftLeg']] = left_knee_contact
    contacts[:,SMPL_JOINTS['rightLeg']] = right_knee_contact

    return offset_floor_height, contacts, discard_seq

def detect_joint_contact(body_joint_seq, joint_name, floor_height, vel_thresh, height_thresh):
    # calc velocity
    joint_seq = body_joint_seq[:, SMPL_JOINTS[joint_name], :]
    joint_vel = np.linalg.norm(joint_seq[1:] - joint_seq[:-1], axis=1)
    joint_vel = np.append(joint_vel, joint_vel[-1])
    # determine contact by velocity
    joint_contact = joint_vel < vel_thresh
    # compute heights
    joint_heights = joint_seq[:, 2] - floor_height
    # compute contact by vel + height
    joint_contact = np.logical_and(joint_contact, joint_heights < height_thresh)

    return joint_contact

def compute_align_mats(root_orient):
    '''   compute world to canonical frame for each timestep (rotation around up axis) '''
    num_frames = root_orient.shape[0]
    # convert aa to matrices
    root_orient_mat = batch_rodrigues(torch.Tensor(root_orient).reshape(-1, 3)).numpy().reshape((num_frames, 9))

    # return compute_world2aligned_mat(torch.Tensor(root_orient_mat).reshape((num_frames, 3, 3))).numpy()

    # rotate root so aligning local body right vector (-x) with world right vector (+x)
    #       with a rotation around the up axis (+z)
    body_right = -root_orient_mat.reshape((num_frames, 3, 3))[:,:,0] # in body coordinates body x-axis is to the left
    world2aligned_mat, world2aligned_aa = compute_align_from_right(body_right)

    return world2aligned_mat

def compute_joint_align_mats(joint_seq):
    '''
    Compute world to canonical frame for each timestep (rotation around up axis)
    from the given joint seq (T x J x 3)
    '''
    left_idx = SMPL_JOINTS['leftUpLeg']
    right_idx = SMPL_JOINTS['rightUpLeg']

    body_right = joint_seq[:, right_idx] - joint_seq[:, left_idx]
    body_right = body_right / np.linalg.norm(body_right, axis=1)[:,np.newaxis]

    world2aligned_mat, world2aligned_aa = compute_align_from_right(body_right)

    return world2aligned_mat

def compute_align_from_right(body_right):
    world2aligned_angle = np.arccos(body_right[:,0] / (np.linalg.norm(body_right[:,:2], axis=1) + 1e-8)) # project to world x axis, and compute angle
    body_right[:,2] = 0.0
    world2aligned_axis = np.cross(body_right, np.array([[1.0, 0.0, 0.0]]))

    world2aligned_aa = (world2aligned_axis / (np.linalg.norm(world2aligned_axis, axis=1)[:,np.newaxis]+ 1e-8)) * world2aligned_angle[:,np.newaxis]
    world2aligned_mat = batch_rodrigues(torch.Tensor(world2aligned_aa).reshape(-1, 3)).numpy()

    return world2aligned_mat, world2aligned_aa

def estimate_velocity(data_seq, h):
    '''
    Given some data sequence of T timesteps in the shape (T, ...), estimates
    the velocity for the middle T-2 steps using a second order central difference scheme.
    - h : step size
    '''
    data_tp1 = data_seq[2:]
    data_tm1 = data_seq[0:-2]
    data_vel_seq = (data_tp1 - data_tm1) / (2*h)
    return data_vel_seq

def estimate_angular_velocity(rot_seq, h):
    '''
    Given a sequence of T rotation matrices, estimates angular velocity at T-2 steps.
    Input sequence should be of shape (T, ..., 3, 3)
    '''
    # see https://en.wikipedia.org/wiki/Angular_velocity#Calculation_from_the_orientation_matrix
    dRdt = estimate_velocity(rot_seq, h)
    R = rot_seq[1:-1]
    RT = np.swapaxes(R, -1, -2)
    # compute skew-symmetric angular velocity tensor
    w_mat = np.matmul(dRdt, RT) 

    # pull out angular velocity vector
    # average symmetric entries
    w_x = (-w_mat[..., 1, 2] + w_mat[..., 2, 1]) / 2.0
    w_y = (w_mat[..., 0, 2] - w_mat[..., 2, 0]) / 2.0
    w_z = (-w_mat[..., 0, 1] + w_mat[..., 1, 0]) / 2.0
    w = np.stack([w_x, w_y, w_z], axis=-1)

    return w

def process_seq(motion_seq_path, babel_id, data_dir, smplh_root):
    # load in input data
    # we leave out "dmpls" and "marker_data"/"marker_label" which are not present in all datasets
    bdata = np.load(motion_seq_path)
    gender = np.array(bdata['gender'], ndmin=1)[0]
    gender = str(gender, 'utf-8') if isinstance(gender, bytes) else str(gender)
    fps = bdata['mocap_framerate']
    num_frames = bdata['poses'].shape[0]
    trans = bdata['trans'][:]               # global translation
    root_orient = bdata['poses'][:, :3]     # global root orientation (1 joint)
    pose_body = bdata['poses'][:, 3:66]     # body joint rotations (21 joints)
    pose_hand = bdata['poses'][:, 66:]      # finger articulation joint rotations
    betas = bdata['betas'][:]               # body shape parameters
    thetas = bdata['poses'][:, get_joints_to_use()] # used for MotionCLIP baseline

    # correct mislabeled data
    if motion_seq_path.find('BMLhandball') >= 0:
        fps = 240
    if motion_seq_path.find('20160930_50032') >= 0 or motion_seq_path.find('20161014_50033') >= 0:
        fps = 59
    original_fps = fps

    # discard if shorter than threshold
    if num_frames < DISCARD_SHORTER_THAN_FRAMES:
        assert False # discarding invalid sequences should be handled in extract_motion_concepts.py L27
    
    # must do SMPL forward pass to get joints
    # split into manageable chunks to avoid running out of GPU memory for SMPL
    body_joint_seq = []
    motionclip_joint_seq = []
    body_vtx_seq = []
    process_inds = [0, min([num_frames, SPLIT_FRAME_LIMIT])]
    while process_inds[0] < num_frames:
        sidx, eidx = process_inds
        body = get_body_model_sequence(smplh_root, gender, process_inds[1] - process_inds[0],
                            pose_body[sidx:eidx], pose_hand[sidx:eidx], betas, root_orient[sidx:eidx], trans[sidx:eidx])
        cur_joint_seq = c2c(body.Jtr)
        cur_body_joint_seq = cur_joint_seq[:, :len(SMPL_JOINTS), :]
        cur_motionclip_joint_seq = cur_joint_seq[:,action2motion_joints,:]
        body_joint_seq.append(cur_body_joint_seq)
        motionclip_joint_seq.append(cur_motionclip_joint_seq)

        # save specific vertices if desired
        if SAVE_KEYPT_VERTS:
            cur_vtx_seq = c2c(body.v)
            cur_mojo_seq = cur_vtx_seq[:,KEYPT_VERTS,:]
            body_vtx_seq.append(cur_mojo_seq)

        process_inds[0] = process_inds[1]
        process_inds[1] = min([num_frames, process_inds[1] + SPLIT_FRAME_LIMIT])
    joint_seq = np.concatenate(body_joint_seq, axis=0)
    motionclip_joint_seq = np.concatenate(motionclip_joint_seq, axis=0)

    vtx_seq = None
    if SAVE_KEYPT_VERTS:
        vtx_seq = np.concatenate(body_vtx_seq, axis=0)

    # determine floor height and foot contacts (fair amount of cpu memeory needed)
    floor_height, contacts, discard_seq = determine_floor_height_and_contacts(joint_seq, fps)
    # translate so floor is at z=0
    trans[:,2] -= floor_height
    joint_seq[:,:,2] -= floor_height
    if SAVE_KEYPT_VERTS:
        vtx_seq[:,:,2] -= floor_height

    # need the joint transform at all steps to find the angular velocity
    joints_world2aligned_rot = compute_joint_align_mats(joint_seq)

    # estimate various velocities based on full frame rate
    #       with second order central difference.
    joint_vel_seq = vtx_vel_seq = trans_vel_seq = root_orient_vel_seq = pose_body_vel_seq = joint_orient_vel_seq = None
    if SAVE_VELOCITIES:
        h = 1.0 / fps
        # joints
        joint_vel_seq = estimate_velocity(joint_seq, h)
        if SAVE_KEYPT_VERTS:
            # vertices
            vtx_vel_seq = estimate_velocity(vtx_seq, h)

        # translation
        trans_vel_seq = estimate_velocity(trans, h)
        # root orient
        root_orient_mat = axisangle2matrots(root_orient.reshape(num_frames, 1, 3)).reshape((num_frames, 3, 3))
        root_orient_vel_seq = estimate_angular_velocity(root_orient_mat, h)
        # body pose
        pose_body_mat = axisangle2matrots(pose_body.reshape(num_frames, len(SMPL_JOINTS)-1, 3)).reshape((num_frames, len(SMPL_JOINTS)-1, 3, 3))
        pose_body_vel_seq = estimate_angular_velocity(pose_body_mat, h)

        # joint up-axis angular velocity (need to compute joint frames first...)
        joint_orient_vel_seq = -estimate_angular_velocity(joints_world2aligned_rot, h)
        # only need around z
        joint_orient_vel_seq = joint_orient_vel_seq[:,2]
        # exit()

        # throw out edge frames for other data so velocities are accurate NOTE: will cause discrepancy with BABEL frame labels
        num_frames = num_frames - 2
        contacts = contacts[1:-1]
        trans = trans[1:-1]
        root_orient = root_orient[1:-1]
        pose_body = pose_body[1:-1]
        pose_hand = pose_hand[1:-1]
        joint_seq = joint_seq[1:-1]
        thetas = thetas[1:-1]
        if SAVE_KEYPT_VERTS:
            vtx_seq = vtx_seq[1:-1]

    # downsample before saving
    if OUT_FPS != fps:
        if OUT_FPS > fps:
            assert False # Cannot supersample data, but all samples should be at 30 fps
        else:
            fps_ratio = float(OUT_FPS) / fps
            new_num_frames = int(fps_ratio*num_frames)
            downsamp_inds = np.linspace(0, num_frames-1, num=new_num_frames, dtype=int)

            # update data to save
            fps = OUT_FPS
            num_frames = new_num_frames
            contacts = contacts[downsamp_inds]
            trans = trans[downsamp_inds]
            root_orient = root_orient[downsamp_inds]
            pose_body = pose_body[downsamp_inds]
            pose_hand = pose_hand[downsamp_inds]
            joint_seq = joint_seq[downsamp_inds]
            thetas = thetas[downsamp_inds]
            motionclip_joint_seq = motionclip_joint_seq[downsamp_inds]
            if SAVE_KEYPT_VERTS:
                vtx_seq = vtx_seq[downsamp_inds]
            
            if SAVE_VELOCITIES:
                joint_vel_seq = joint_vel_seq[downsamp_inds]
                if SAVE_KEYPT_VERTS:
                    vtx_vel_seq = vtx_vel_seq[downsamp_inds]
                trans_vel_seq = trans_vel_seq[downsamp_inds]
                root_orient_vel_seq = root_orient_vel_seq[downsamp_inds]
                pose_body_vel_seq = pose_body_vel_seq[downsamp_inds]

                # joint up-axis angular velocity (need to compute joint frames first...)
                joint_orient_vel_seq = joint_orient_vel_seq[downsamp_inds]

    world2aligned_rot = None
    if SAVE_ALIGN_ROT:
        # compute rotation to canonical frame (forward facing +y) for every frame
        world2aligned_rot = compute_align_mats(root_orient)

    if discard_seq:
        assert False # Discarding invalid sequences should be handled in extract_motion_concepts.py L27

    out_dir = osp.join(data_dir, 'motion_sequences', babel_id)

    if not osp.exists(out_dir):
        os.makedirs(out_dir)
    
    # save pose (general use)
    pose_file_path = osp.join(out_dir, 'pose.npz')
    np.savez(pose_file_path, fps=fps,
                               mocap_framerate=original_fps,
                               gender=str(gender),
                               floor_height=floor_height,
                               contacts=contacts,
                               trans=trans,
                               root_orient=root_orient,
                               pose_body=pose_body,
                               pose_hand=pose_hand,
                               betas=betas,
                               joints=joint_seq,
                               mojo_verts=vtx_seq,
                               joints_vel=joint_vel_seq,
                               mojo_verts_vel=vtx_vel_seq,
                               trans_vel=trans_vel_seq,
                               root_orient_vel=root_orient_vel_seq,
                               joint_orient_vel_seq=joint_orient_vel_seq,
                               pose_body_vel=pose_body_vel_seq,
                               world2aligned_rot=world2aligned_rot,
                               thetas=thetas,
                               motionclip_joint_seq=motionclip_joint_seq)

    # save processed joints (NSPose representation)
    processed_joints = joint_seq
    for coordinate in range(0, 2): # normalize x, y component of all poses by the first pose's root joint position
        processed_joints[:, :, coordinate] = processed_joints[:, :, coordinate] - processed_joints[0, 0, coordinate]
    global_world2aligned_rot = compute_joint_align_mats(processed_joints)[0]
    processed_joints = np.matmul(global_world2aligned_rot, processed_joints.reshape((-1, 3)).T).T.reshape((np.shape(processed_joints)[0], len(SMPL_JOINTS), 3))

    joints_file_path = osp.join(out_dir, 'joints')
    np.save(joints_file_path, processed_joints)

    # save rots6d rep (MotionClip representation)
    motionclip_joints3D = motionclip_joint_seq - motionclip_joint_seq[0, 0, :]
    ret = torch.from_numpy(motionclip_joints3D)
    ret_tr = ret[:, 0, :]

    rot_convention = 'legacy'
    poses = thetas.reshape(-1, ROT_CONVENTION_TO_ROT_NUMBER[rot_convention] + 1,
                                                     3)  # +1 for global orientation
    ret = matrix_to_rotation_6d(axis_angle_to_matrix(torch.from_numpy(poses)))

    padded_tr = torch.zeros((ret.shape[0], ret.shape[2]), dtype=ret.dtype)
    padded_tr[:, :3] = ret_tr
    ret = torch.cat((ret, padded_tr[:, None]), 1)
    motionclip_rep = ret.permute(1, 2, 0).contiguous().float().numpy()

    rots6d_file_path = osp.join(out_dir, 'rots6d')
    np.save(rots6d_file_path, motionclip_rep)

def get_amass_file_path(motion_seq_path, amass_root):
    motion_seq_path = osp.join(amass_root, *(motion_seq_path.split(osp.sep)[1:])) # remove parent folder
    assert osp.isfile(motion_seq_path)
    return motion_seq_path

def process_amass_data(data_dir, babel_root, amass_root, smplh_root):
    questions = json.load(open(osp.join(data_dir, 'questions.json')))

    babel_ids = []
    for _, question in questions.items():
        babel_ids.append(question['babel_id'])
    
    babel_motion_seq_paths = {}
    for spl in ['train', 'val']:
        ann = json.load(open(osp.join(babel_root, f'{spl}.json')))
        ann = {babel_id: ann[babel_id]['feat_p'] for babel_id in ann}
        babel_motion_seq_paths.update(ann)

    for babel_id in tqdm(babel_ids):
        motion_seq_path = babel_motion_seq_paths[babel_id]
        motion_seq_path = get_amass_file_path(motion_seq_path, amass_root)
        process_seq(motion_seq_path, babel_id, data_dir, smplh_root)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Root directory of BABEL-QA dataset')
    parser.add_argument('--babel_root', type=str, required=True, help='Root directory of the BABEL dataset')
    parser.add_argument('--amass_root', type=str, required=True, help='Root directory of raw AMASS dataset.')
    parser.add_argument('--smplh_root', type=str, required=True, help='Root directory of the SMPL+H body model.')
    args = parser.parse_args()

    process_amass_data(args.data_dir, args.babel_root, args.amass_root, args.smplh_root)


if __name__ == "__main__":
    main()