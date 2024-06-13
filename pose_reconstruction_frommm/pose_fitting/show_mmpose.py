import torch
import os
import os.path as osp
import trimesh
import numpy as np
import smplx
from smplx import SMPL, SMPLH, SMPLX
# import torchgeometry

# from common import constants

import time

import sys
sys.path.append("./")
sys.path.append("./mano_tools")
# print('__file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(
#     __file__, __name__, str(__package__)))

# from util_rot import matrix_to_axis_angle
from utils.utils import extract_opticdata, loadbodypose_humor
from utils.util_rot import *
from mano_tools.extract_hands_mano import extract_hands_manopose
from human_tools.body_model import BodyModel
import platform

print(os.getcwd())

# ============ config ===========================
root_folder = "F:/data/throw"
if platform.system().lower() != "windows":
    root_folder = "/media/ur-5/lx_assets/data/throw"
# img_dir = "D:/JustLX/data/011998/processed/rgbd0"
# save_results = True 
save_meshes = True
stage = "throw"


# ===============================================

def main(file_path, device):
    
    # folders
    file_path = file_path.replace("\\","/")
    basename = file_path.split('/')[-1]
    meshes_dir = file_path +  "_body_meshes"
    if not os.path.exists(meshes_dir):
        os.makedirs(meshes_dir)
    
    # load body pose 
    results_out = meshes_dir
    body_file = file_path
    smpl_para = np.load(body_file)
    smpl_poses = smpl_para['pose_body'] # seq_len*63
    smpl_hand = smpl_para['pose_hand'] # seq_len*[45+45]
    n = smpl_poses.shape[0] 
    betas = smpl_para['betas'] # 16
    smpl_transl = smpl_para['trans'].reshape(n,-1) # seq_len*3
    root_orient = smpl_para['root_orient']
    
    mm_file = "X:\\Just\\OneDrive\\0_MyProjects\\RoboticsX\\H2TC_Dataset\\HumanPoseExtraction\\mmhuman3d\\vis_results\\002870_spin.npz"
    mm_para = np.load(mm_file, allow_pickle=True)
    mm_smpl = mm_para['smpl']
    mm_verts = mm_para['verts']
    mm_person_id = mm_para['person_id']
    fl = mm_smpl.flat
    for value in fl:
        # print(key)
        mm_pose= value['body_pose']
        # mm_orint = 
        # mm_transl = 
        # mm_pose = value['body_pose'][mm_person_id == 1][:,:-2,:] # [frame, 21 joints, 3]

    # load smplh model 
    SMPLH_HUMOR_MODEL = "./human_tools/smplh_humor/male/model.npz"
    smplh_model = BodyModel(SMPLH_HUMOR_MODEL, \
         num_betas=16, \
        batch_size = smpl_poses.shape[0],\
            ).to("cuda")
    
    # load smplh model  
    SMPL_MODEL = "./human_tools/smpl/SMPL_MALE.pkl"
    smpl_mm = BodyModel(SMPL_MODEL, \
        #  num_betas=10, \
        batch_size = smpl_poses.shape[0],\
            model_type='smpl',\
            ).to("cuda")


    smpl_poses = torch.from_numpy(smpl_poses).to(torch.float32).to("cuda")
    hand_poses = torch.from_numpy(smpl_hand).to(torch.float32).to("cuda")
    root_orient = torch.from_numpy(root_orient).to(torch.float32).to("cuda")
    smpl_transl = torch.from_numpy(smpl_transl).to(torch.float32).to("cuda")
    
    # # output persons [our pose + our global]
    with torch.no_grad():
        pred_output = smplh_model.bm(
                                body_pose=smpl_poses,
                                global_orient=root_orient,
                                left_hand_pose=hand_poses[:,:45],
                                right_hand_pose=hand_poses[:,45:],
                                transl=smpl_transl)
    verts = pred_output.vertices.cpu().numpy()
    faces = smplh_model.bm.faces

    # output persons [ours]
    # if save_meshes:
    #     print(f"save meshes to \"{meshes_dir}\"")
    #     os.makedirs(meshes_dir, exist_ok=True)
        
    #     n = len(verts)
    #     id = 0
    #     for ii in range(n):
    #         verts0 = np.array(verts[ii])
    #         mesh0 = trimesh.Trimesh(verts0, faces)
                
    #         # save mesh0
    #         fram_name =  str(0 + ii)
    #         filename =  "our_body_%s.ply" % (fram_name)                                                            
    #         out_mesh_path = osp.join(meshes_dir, filename)
    #         mesh0.export(out_mesh_path)
    
    # # output persons [mm_pose + our global]
    # for i in range(3):
    #     # transfer the mmhuman to our pose
    #     n = smpl_poses.shape[0]
    #     t_pose = mm_pose[mm_person_id == i][:n,:-2,:].reshape((n,63))
    #     smpl_poses = torch.from_numpy(t_pose).to(torch.float32).to("cuda")
            

    #     with torch.no_grad():
    #         pred_output = smplh_model.bm(
    #                                 body_pose=smpl_poses,
    #                                 global_orient=root_orient,
    #                                 left_hand_pose=hand_poses[:,:45],
    #                                 right_hand_pose=hand_poses[:,45:],
    #                                 transl=smpl_transl)
    #     verts = pred_output.vertices.cpu().numpy()
    #     faces = smplh_model.bm.faces

    #     if save_meshes:
    #         print(f"save meshes to \"{meshes_dir}\"")
    #         os.makedirs(meshes_dir, exist_ok=True)
            
    #         n = len(verts)
    #         id = 0
    #         for ii in range(n):
    #             verts0 = np.array(verts[ii])
    #             mesh0 = trimesh.Trimesh(verts0, faces)
                    
    #             # save mesh0
    #             fram_name =  str(0 + ii)
    #             filename =  "id_%s_humor_body_%s.ply" % (i,fram_name)                                                            
    #             out_mesh_path = osp.join(meshes_dir, filename)
    #             mesh0.export(out_mesh_path)
    #             a = 1
    
    # # output persons [mm_pose and global RT]
    meshes_dir = os.path.join(meshes_dir,'mmhuman_orig_smpl')
    n = smpl_poses.shape[0]
    for i in range(3):
        # transfer the mmhuman to our pose
        t_pose = mm_pose[mm_person_id == i][:n,:,:].reshape((n,69))
        
        smpl_poses = torch.from_numpy(t_pose).to(torch.float32).to("cuda")
        # root_orient = torch.from_numpy(mm_orint[mm_person_id == i]).to(torch.float32).to("cuda")
        # smpl_transl = torch.from_numpy(mm_transl[mm_person_id == i]).to(torch.float32).to("cuda")    

        with torch.no_grad():
            pred_output = smpl_mm.bm(
                                    body_pose=smpl_poses)
        verts = pred_output.vertices.cpu().numpy()
        faces = smpl_mm.bm.faces

        if save_meshes:
            print(f"save meshes to \"{meshes_dir}\"")
            os.makedirs(meshes_dir, exist_ok=True)
            
            n = len(verts)
            id = 0
            for ii in range(n):
                verts0 = np.array(verts[ii])
                mesh0 = trimesh.Trimesh(verts0, faces)
                    
                # save mesh0
                fram_name =  str(0 + ii)
                filename =  "id_%s_humor_body_%s.ply" % (i,fram_name)                                                            
                out_mesh_path = osp.join(meshes_dir, filename)
                mesh0.export(out_mesh_path)
                a = 1
           
"""
print("--------------------------- Visualization ---------------------------")
# make the output directory
os.makedirs(front_view_dir, exist_ok=True)
print("Front view directory:", front_view_dir)
if show_sideView:
    os.makedirs(side_view_dir, exist_ok=True)
    print("Side view directory:", side_view_dir)
if show_bbox:
    os.makedirs(bbox_dir, exist_ok=True)
    print("Bounding box directory:", bbox_dir)

pred_vert_arr = np.array(pred_vert_arr)
for img_idx, orig_img_bgr in enumerate(tqdm(orig_img_bgr_all)):
    chosen_mask = detection_all[:, 0] == img_idx
    chosen_vert_arr = pred_vert_arr[chosen_mask]

    # setup renderer for visualization
    img_h, img_w, _ = orig_img_bgr.shape
    focal_length = estimate_focal_length(img_h, img_w)
    renderer = Renderer(focal_length=focal_length, img_w=img_w, img_h=img_h,
                        faces=smplh_model.faces,
                        same_mesh_color=True)
    front_view = renderer.render_front_view(chosen_vert_arr,
                                            bg_img_rgb=orig_img_bgr[:, :, ::-1].copy())

    # save rendering results
    basename = osp.basename(img_path_list[img_idx]).split(".")[0]
    filename = basename + "_front_view_cliff_%s.jpg" % backbone
    front_view_path = osp.join(front_view_dir, filename)
    cv2.imwrite(front_view_path, front_view[:, :, ::-1])

    if show_sideView:
        side_view_img = renderer.render_side_view(chosen_vert_arr)
        filename = basename + "_side_view_cliff_%s.jpg" % backbone
        side_view_path = osp.join(side_view_dir, filename)
        cv2.imwrite(side_view_path, side_view_img[:, :, ::-1])

    # delete the renderer for preparing a new one
    renderer.delete()

    # draw the detection bounding boxes
    if show_bbox:
        chosen_detection = detection_all[chosen_mask]
        bbox_info = chosen_detection[:, 1:6]

        bbox_img_bgr = orig_img_bgr.copy()
        for min_x, min_y, max_x, max_y, conf in bbox_info:
            ul = (int(min_x), int(min_y))
            br = (int(max_x), int(max_y))
            cv2.rectangle(bbox_img_bgr, ul, br, color=(0, 255, 0), thickness=2)
            cv2.putText(bbox_img_bgr, "%.1f" % conf, ul,
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1.0, color=(0, 0, 255), thickness=1)
        filename = basename + "_bbox.jpg"
        bbox_path = osp.join(bbox_dir, filename)
        cv2.imwrite(bbox_path, bbox_img_bgr)

# make videos
if make_video:
    print("--------------------------- Making videos ---------------------------")
    from common.utils import images_to_video
    images_to_video(front_view_dir, video_path=front_view_dir + ".mp4", frame_rate=frame_rate)
    if show_sideView:
        images_to_video(side_view_dir, video_path=side_view_dir + ".mp4", frame_rate=frame_rate)
    if show_bbox:
        images_to_video(bbox_dir, video_path=bbox_dir + ".mp4", frame_rate=frame_rate)
"""    

              

if __name__ == "__main__":
    
    # == human detection model  ==
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
       
    file = "L:\\002870_104_frames_60_fps.npz"
    main(file, device)
        
            
    



