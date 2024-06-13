import os
import subprocess
import shutil
import numpy as np
import sys, os
cur_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_file_path, '..'))
sys.path.append(os.path.join(cur_file_path, '.'))
# from fitting_utils import vis_results_twosubjs

root_folders = [
    "D:\\h2tc_dataset",
                ]
error_info = {}

# camextr: numpy arr [4,4]
def isgoodcamextrix(camextr):
    if np.isnan(camextr).any():
        return False
    if (camextr[0,3]<0) & (camextr[1,3]>0) & (camextr[2,3]>0) or (camextr[0,3]>0) & (camextr[1,3]<0) & (camextr[2,3]<0):
        return True
    else:
        return False 


for root_folder in root_folders:
    takes = os.listdir(root_folder)
    takes.sort()
    # takes.reverse()
    for take in takes:
        rgbd = "rgbd0"
        #### run left sub1 person:  ###
        take_folder = root_folder + '/' + take
        data_path = take_folder + '/processed/'+rgbd
        if not os.path.exists(data_path):
            continue
        
        out_folder = os.path.join(root_folder, take, 'processed/rgbd0/humor_out_neutral')
        out_folder = out_folder.replace("\\","/")
        
        anno_file = os.path.join(take_folder, take+".json")
        if os.path.exists(anno_file)==False:
            error_info[take] = "no annotation file."
            continue
        
        try:
            # if os.path.exists(out_folder + "/results_out/stage2_results.npz"):
            # #     camextr = np.loadtxt(take_folder+'/CamExtr.txt')
            # #     if not isgoodcamextrix(camextr):
            # #         print("CAM ERROR: %s" % take_folder)
            # #     else:
            #     continue
            os.makedirs(out_folder, exist_ok=True)
            
            # src_dir = os.path.join(root_folder, take, 'processed/rgbd0/humor_out_v2/rgb_preprocess')
            # tar_dir = out_folder + '/rgb_preprocess'
            # if os.path.exists(src_dir) and (not os.path.exists(tar_dir)):
            #     shutil.copytree(src_dir, tar_dir)
            
            # flag = "throw" if "throw" in root_folder else "catch"
            flag = "catch"
            
            if not os.path.exists(os.path.join(data_path,'humor_out_neutral/results_out/stage2_results.npz')):
            # if True:
                run_cmds = ["python", "pose_fitting/run_fitting.py", \
                    "@./config/fitting/fit_h2tc.cfg", \
                    "--data-path", data_path , \
                    "--out" ,  out_folder, \
                        "--is-sub1","sub1",  \
                            "--catch-throw", flag, \
                                # "--vis", "--num_frame", "200"
                        ]
                print("***********", run_cmds, "*****************")
                subprocess.run(run_cmds)
            
            # # # for sub2
            # if not os.path.exists(os.path.join(data_path,'humor_out_v3_sub2/results_out/stage2_results.npz')):
            #     out_folder = os.path.join(root_folder, take, 'processed/rgbd0/humor_out_v3_sub2')
            #     run_cmds = ["python", "humor/fitting/run_fitting.py", \
            #     "@./configs/fit_rgb_demo_lx_nosplit.cfg", \
            #     "--num-iters", "10", "40", "20", \
            #     "--data-path", data_path , \
            #     "--out" ,  out_folder, \
            #         "--is-sub1","sub2",  \
            #             "--catch-throw", flag, \
            #                 "--vis",\
            #                 "--num_frame", "200"
            #         ]
            #     print("***********", run_cmds, "*****************")
            #     subprocess.run(run_cmds)
            
            # # viz two subjects in the same view
            # if not os.path.exists(os.path.join(data_path,'humor_out_two_subjs/humor_out_two_subjects.mp4')):
            # # if True:
            #     print("*********** VISUALIZATION *****************")
            #     vis_results_twosubjs(take_folder)
            
        except Exception as e:
            print("ERROR Take %s: %s",(take, str(e)))
            error_info[take] = str(e)
            
        # #### run right sub2 person:  ###
        # take_folder = root_folder + '/' + take
        # data_path = take_folder + '/processed/'+rgbd
        # out_folder = data_path + '/humor_out_v2_sub2'
        
        # try:
        #     if os.path.exists(out_folder + "/results_out"):
        #         continue
        #     os.makedirs(out_folder, exist_ok=True)
            
        #     src_dir = data_path + '/humor_out_v2' + '/rgb_preprocess'
        #     tar_dir = out_folder + '/rgb_preprocess'
        #     if os.path.exists(src_dir):
        #         shutil.move(src_dir, tar_dir)
            
        #     run_cmds = ["python", "humor/fitting/run_fitting.py", \
        #         "@./configs/fit_rgb_demo_lx_nosplit.cfg", \
        #         "--num-iters", "10", "50", "20", \
        #         "--data-path", data_path , \
        #         "--out" ,  out_folder, \
        #             "--is-sub1","sub2"]
            
        #     print("***********", run_cmds, "*****************")
        #     subprocess.run(run_cmds)
        # except Exception as e:
        #     print("ERROR Take %s: %s",(take, str(e)))
        #     error_info[take] = str(e)
    print("Error info: ------------------------------------------------------")
    for take in error_info.keys():
        print("ERROR Take" ,take, ":", error_info[take])    
    
                
print("Error info: ------------------------------------------------------")
for take in error_info.keys():
    print("ERROR Take" ,take, ":", error_info[take])
        