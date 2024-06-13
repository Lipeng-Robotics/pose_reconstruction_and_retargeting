import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pylab as pylab
import os
import json
import sys
from sklearn.cluster import DBSCAN
import platform
import time
import glob
import trimesh
import torch
import argparse

from util_tool.viz import viz_smpl_seq

sys.path.append('..')
from utils.util_rot import batch_rodrigues, axisangle2matrots
from utils.utils import loadwholebodypose
from human_tools.body_model import BodyModel, smpl_connections,SMPL_JOINTS,SMPLH_PATH
from human_tools.mesh_viewer import MeshViewer, colors
#
# Processing options
#

OUT_FPS = 60
SAVE_KEYPT_VERTS = True # save vertex locations of certain keypoints
SAVE_HAND_POSE = True # save joint angles for the hand
SAVE_VELOCITIES = True # save all parameter velocities available
SAVE_ALIGN_ROT = True # save rot mats that go from world root orient to aligned root orient
DISCARD_TERRAIN_SEQUENCES = True # throw away sequences where the person steps onto objects (determined by a heuristic)

# optional viz during processing
VIZ_PLOTS = False
VIZ_SEQ = False

# if sequence is longer than this, splits into sequences of this size to avoid running out of memory
# ~ 4000 for 12 GB GPU, ~2000 for 8 GB
SPLIT_FRAME_LIMIT = 2000

NUM_BETAS = 16 # size of SMPL shape parameter to use

DISCARD_SHORTER_THAN = 1.0 # seconds

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

c2c = lambda tensor: tensor.detach().cpu().numpy()



def get_body_model_sequence(smplh_path, gender, num_frames,
                  pose_body, pose_hand, betas, root_orient, trans):
    gender = str(gender)
    bm_path = os.path.join(smplh_path, gender + '/smplh_%s.npz'%gender)
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

    if VIZ_PLOTS:
        fig = plt.figure()
        steps = np.arange(num_frames)
        plt.plot(steps, left_toe_vel, '-r', label='left vel')
        plt.plot(steps, right_toe_vel, '-b', label='right vel')
        plt.legend()
        plt.show()
        plt.close()

    # now foot heights (y is up)
    left_toe_heights = left_toe_seq[:, 1]
    right_toe_heights = right_toe_seq[:, 1]
    root_heights = root_seq[:, 1]

    if VIZ_PLOTS:
        fig = plt.figure()
        steps = np.arange(num_frames)
        plt.plot(steps, left_toe_heights, '-r', label='left toe height')
        plt.plot(steps, right_toe_heights, '-b', label='right toe height')
        plt.plot(steps, root_heights, '-g', label='root height')
        plt.legend()
        plt.show()
        plt.close()

    # filter out heights when velocity is greater than some threshold (not in contact)
    all_inds = np.arange(left_toe_heights.shape[0])
    left_static_foot_heights = left_toe_heights[left_toe_vel < FLOOR_VEL_THRESH]
    left_static_inds = all_inds[left_toe_vel < FLOOR_VEL_THRESH]
    right_static_foot_heights = right_toe_heights[right_toe_vel < FLOOR_VEL_THRESH]
    right_static_inds = all_inds[right_toe_vel < FLOOR_VEL_THRESH]

    all_static_foot_heights = np.append(left_static_foot_heights, right_static_foot_heights)
    all_static_inds = np.append(left_static_inds, right_static_inds)

    if VIZ_PLOTS:
        fig = plt.figure()
        steps = np.arange(left_static_foot_heights.shape[0])
        plt.plot(steps, left_static_foot_heights, '-r', label='left static height')
        plt.legend()
        plt.show()
        plt.close()

    # fig = plt.figure()
    # plt.hist(all_static_foot_heights)
    # plt.show()
    # plt.close()

    discard_seq = False
    if all_static_foot_heights.shape[0] > 0:
        cluster_heights = []
        cluster_root_heights = []
        cluster_sizes = []
        # cluster foot heights and find one with smallest median
        clustering = DBSCAN(eps=0.005, min_samples=3).fit(all_static_foot_heights.reshape(-1, 1))
        all_labels = np.unique(clustering.labels_)
        # print(all_labels)
        if VIZ_PLOTS:
            plt.figure()
        min_median = min_root_median = float('inf')
        for cur_label in all_labels:
            cur_clust = all_static_foot_heights[clustering.labels_ == cur_label]
            cur_clust_inds = np.unique(all_static_inds[clustering.labels_ == cur_label]) # inds in the original sequence that correspond to this cluster
            if VIZ_PLOTS:
                plt.scatter(cur_clust, np.zeros_like(cur_clust), label='foot %d' % (cur_label))
            # get median foot height and use this as height
            cur_median = np.median(cur_clust)
            cluster_heights.append(cur_median)
            cluster_sizes.append(cur_clust.shape[0])

            # get root information
            cur_root_clust = root_heights[cur_clust_inds]
            cur_root_median = np.median(cur_root_clust)
            cluster_root_heights.append(cur_root_median)
            if VIZ_PLOTS:
                plt.scatter(cur_root_clust, np.zeros_like(cur_root_clust), label='root %d' % (cur_label))

            # update min info
            if cur_median < min_median:
                min_median = cur_median
                min_root_median = cur_root_median

        # print(cluster_heights)
        # print(cluster_root_heights)
        # print(cluster_sizes)
        if VIZ_PLOTS:
            plt.show()
            plt.close()

        floor_height = min_median 
        offset_floor_height = floor_height - FLOOR_HEIGHT_OFFSET # toe joint is actually inside foot mesh a bit

        if DISCARD_TERRAIN_SEQUENCES:
            # print(min_median + TERRAIN_HEIGHT_THRESH)
            # print(min_root_median + ROOT_HEIGHT_THRESH)
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
    world2aligned_mat, world2aligned_aa = compute_align_from_right_lx(body_right)

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

def compute_align_from_right_lx(body_right):
    xz = np.concatenate([body_right[:,0:1],body_right[:,1:2]], axis=1)
    world2aligned_angle = np.arccos(body_right[:,0] / (np.linalg.norm(xz, axis=1) + 1e-8)) # project to world x axis, and compute angle
    body_right[:,1] = 0.0
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

def process_seq(m_data, output_file_path):
   
    start_t = time.time()

    print(output_file_path)

    smpl_transl,root_orient,wholebody_pose_seq,hand_label,s_frame, e_frame, betas = m_data
    gender = 'male'
    gender = str(gender, 'utf-8') if isinstance(gender, bytes) else str(gender)
    fps = 60.0
    num_frames = e_frame - s_frame + 1
    trans = smpl_transl               # global translation
    root_orient = root_orient     # global root orientation (1 joint)
    pose_body = wholebody_pose_seq[:, :63]     # body joint rotations (21 joints)
    pose_hand = wholebody_pose_seq[:, 63:]      # finger articulation joint rotations
    betas = betas              # body shape parameters

    print(gender)
    print('fps: %d' % (fps))
    print(trans.shape)
    print(root_orient.shape)
    print(pose_body.shape)
    print(pose_hand.shape)
    print(betas.shape)

    # # discard if shorter than threshold
    # if num_frames < DISCARD_SHORTER_THAN*fps:
    #     print('Sequence shorter than %f s, discarding...' % (DISCARD_SHORTER_THAN))
    #     return

    # must do SMPL forward pass to get joints
    # split into manageable chunks to avoid running out of GPU memory for SMPL
    body_joint_seq = []
    process_inds = [0, min([num_frames, SPLIT_FRAME_LIMIT])]
    while process_inds[0] < num_frames:
        print(process_inds)
        sidx, eidx = process_inds
        body = get_body_model_sequence(SMPLH_PATH, gender, process_inds[1] - process_inds[0],
                            pose_body[sidx:eidx], pose_hand[sidx:eidx], betas, root_orient[sidx:eidx], trans[sidx:eidx])
        cur_joint_seq = c2c(body.Jtr)
        cur_body_joint_seq = cur_joint_seq[:, :len(SMPL_JOINTS), :]
        body_joint_seq.append(cur_body_joint_seq)

        process_inds[0] = process_inds[1]
        process_inds[1] = min([num_frames, process_inds[1] + SPLIT_FRAME_LIMIT])

    joint_seq = np.concatenate(body_joint_seq, axis=0)
    print(joint_seq.shape)

    # TODO：floor?
    # # determine floor height and foot contacts # lx: for me, contact is unnecessary
    floor_height, contacts, discard_seq = determine_floor_height_and_contacts(joint_seq, fps)
    print('Floor height: %f' % (floor_height))
    # lx: translate floor # translate so floor is at y=0
    trans[:,1] -= floor_height
    joint_seq[:,:,1] -= floor_height

    # estimate various velocities based on full frame rate
    #       with second order central difference.
    joint_vel_seq = vtx_vel_seq = trans_vel_seq = root_orient_vel_seq = pose_body_vel_seq = None
    if SAVE_VELOCITIES:
        h = 1.0 / fps
        # joints
        joint_vel_seq = estimate_velocity(joint_seq, h)

        # translation
        trans_vel_seq = estimate_velocity(trans, h)
        # root orient
        root_orient_mat = axisangle2matrots(root_orient.reshape(num_frames, 1, 3)).reshape((num_frames, 3, 3))
        root_orient_vel_seq = estimate_angular_velocity(root_orient_mat, h)

        # throw out edge frames for other data so velocities are accurate
        num_frames = num_frames - 2
        contacts = contacts[1:-1]
        trans = trans[1:-1]
        root_orient = root_orient[1:-1]
        pose_body = pose_body[1:-1]
        pose_hand = pose_hand[1:-1]
        joint_seq = joint_seq[1:-1]

    # downsample before saving
    if OUT_FPS != fps:
        if OUT_FPS > fps:
            print('Cannot supersample data, saving at data rate!')
        else:
            fps_ratio = float(OUT_FPS) / fps
            print('Downsamp ratio: %f' % (fps_ratio))
            new_num_frames = int(fps_ratio*num_frames)
            print('Downsamp num frames: %d' % (new_num_frames))
            # print(cur_num_frames)
            # print(new_num_frames)
            downsamp_inds = np.linspace(0, num_frames-1, num=new_num_frames, dtype=int)
            # print(downsamp_inds)

            # update data to save
            fps = OUT_FPS
            num_frames = new_num_frames
            contacts = contacts[downsamp_inds]
            trans = trans[downsamp_inds]
            root_orient = root_orient[downsamp_inds]
            pose_body = pose_body[downsamp_inds]
            pose_hand = pose_hand[downsamp_inds]
            joint_seq = joint_seq[downsamp_inds]
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
        # compute rotation to canonical frame (forward facing +z) for every frame
        world2aligned_rot = compute_align_mats(root_orient)

    # NOTE: debug viz
    if VIZ_SEQ:
        body = get_body_model_sequence(SMPLH_PATH, gender, num_frames,
                            pose_body, pose_hand, betas, root_orient, trans)    
        # debug_viz_seq(body, fps, contacts=contacts)
        viz_smpl_seq(body, imw=1080, imh=1080, fps=fps, contacts=contacts,
            render_body=True, render_joints=True, render_skeleton=True, render_ground=True,body_alpha=0.5,
            joints_seq=joint_seq) #,
            # joints_vel=root_orient_vel_seq.reshape((-1, 1, 3)).repeat(22, axis=1))
            # points_seq=vtx_seq,
            # points_vel_seq=vtx_vel_seq)
            # root_orient_vel_seq.reshape((-1, 1, 3)).repeat(22, axis=1)

    if not SAVE_HAND_POSE:
        pose_hand = None

    # save
    # add number of frames and framrate to file path for each of loading
    out_folder = os.path.dirname(output_file_path)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    output_file_path = output_file_path[:-4] + '_%d_frames_%d_fps.npz' % (num_frames, int(fps))
    np.savez(output_file_path, fps=fps,
                               gender=str(gender),
                               floor_height=floor_height,
                               contacts=contacts,
                               trans=trans,
                               root_orient=root_orient,
                               pose_body=pose_body,
                               pose_hand=pose_hand,
                               betas=betas,
                               joints=joint_seq,
                            #    mojo_verts=vtx_seq,
                               joints_vel=joint_vel_seq,
                            #    mojo_verts_vel=vtx_vel_seq,
                               trans_vel=trans_vel_seq,
                               root_orient_vel=root_orient_vel_seq,
                            #    joint_orient_vel_seq=joint_orient_vel_seq,
                            #    pose_body_vel=pose_body_vel_seq,
                               world2aligned_rot=world2aligned_rot
                               )

    print('Seq process time: %f s' % (time.time() - start_t))

def main(config):
    start_time = time.time()
    out_folder = config.out
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    # get all available datasets
    all_dataset_dirs = [os.path.join(config.amass_root, f) for f in sorted(os.listdir(config.amass_root)) if f[0] != '.']
    all_dataset_dirs = [f for f in all_dataset_dirs if os.path.isdir(f)]
    print('Found %d available datasets from raw AMASS data source.' % (len(all_dataset_dirs)))
    all_dataset_names = [f.split('/')[-1] for f in all_dataset_dirs]
    print(all_dataset_names)

    # requested datasets
    dataset_dirs = [os.path.join(config.amass_root, f) for f in config.datasets]
    dataset_names = config.datasets
    print('Requested datasets:')
    print(dataset_dirs)
    print(dataset_names)

    # go through each dataset to set up directory structure before processing
    all_seq_in_files = []
    all_seq_out_files = []
    for data_dir, data_name in zip(dataset_dirs, dataset_names):
        if not os.path.exists(data_dir):
            print('Could not find dataset %s in available raw AMASS data!' % (data_name))
            return

        cur_output_dir = os.path.join(out_folder, data_name)
        if not os.path.exists(cur_output_dir):
            os.mkdir(cur_output_dir)

        # first create subject structure in output
        cur_subject_dirs = [f for f in sorted(os.listdir(data_dir)) if f[0] != '.' and os.path.isdir(os.path.join(data_dir, f))]
        print(cur_subject_dirs)
        for subject_dir in cur_subject_dirs:
            cur_subject_out = os.path.join(cur_output_dir, subject_dir)
            if not os.path.exists(cur_subject_out):
                os.mkdir(cur_subject_out)

        # then collect all sequence input files
        input_seqs = glob.glob(os.path.join(data_dir, '*/*_poses.npz'))
        print(len(input_seqs))

        # and create output sequence file names
        output_file_names = ['/'.join(f.split('/')[-2:]) for f in input_seqs]
        output_seqs = [os.path.join(cur_output_dir, f) for f in output_file_names]
        print(len(output_seqs))

        already_processed = [i for i in range(len(output_seqs)) if len(glob.glob(output_seqs[i][:-4] + '*.npz')) == 1]
        already_processed_output_names =  [output_file_names[i] for i in already_processed]
        print('Already processed these sequences, skipping:')
        print(already_processed_output_names)
        not_already_processed = [i for i in range(len(output_seqs)) if len(glob.glob(output_seqs[i][:-4] + '*.npz')) == 0]
        input_seqs = [input_seqs[i] for i in not_already_processed]
        output_seqs = [output_seqs[i] for i in not_already_processed]

        all_seq_in_files += input_seqs
        all_seq_out_files += output_seqs
    
    smplh_paths = [config.smplh_root]*len(all_seq_in_files)
    data_paths = list(zip(all_seq_in_files, all_seq_out_files, smplh_paths))

    for data_in in data_paths:
        process_seq(data_in)

    total_time = time.time() - start_time
    print('TIME TO PROCESS: %f min' % (total_time / 60.0))

if __name__ == "__main__":

    # data folders
    data_folders = ["/media/ur-5/golden_t/data/throw",
                "/media/ur-5/lx_assets/data/throw"]
    out_root = "/home/ur-5/Projects/DATA_processed_motion"
    stage="throw"
    
    if not os.path.exists(out_root):
        os.makedirs(out_root)
    
    # 1. get all motions 
    start_time = time.time()
    error = {}
    # go through each dataset to set up directory structure before processing
    all_seq_in_files = []
    all_seq_out_files = []
    for data_folder in data_folders:
        takes = os.listdir(data_folder)
        takes.sort()
        for take_id in takes:
            try: 
                # load extracted motion
                data_pose = loadwholebodypose(data_folder, take_id, stage=stage)
                if data_pose == None:
                    error[take_id] = 'None\n'
                    continue
                # 
                m_data = data_pose
                
                # filter out wrong takes
                if np.isnan(m_data[2]).any():
                    error[take_id] = 'Nan\n'
                    continue
                if (m_data[0]>10).any() or (m_data[1]>10).any() or (m_data[2]>10).any() :
                    error[take_id] = 'Nan\n'
                    continue
                
                out_folder = os.path.join(out_root, stage, take_id)
                output_file_path = os.path.join(out_folder, "%s.npz" % take_id)
                # 2. process them (velocities, joints, states)
                process_seq(m_data, output_file_path)        

            except Exception as e:
                print(str(e))
                error[take_id] = str(e) + '\n'
    print(error)  
    total_time = time.time() - start_time
    print('TIME TO PROCESS: %f min' % (total_time / 60.0))          



# velocity and state 