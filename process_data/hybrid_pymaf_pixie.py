import os, sys
import argparse
import torch
import numpy as np
import pickle

sys.path.insert(0, './submodules/PIXIE')
from pixielib.utils import util

def main(args):
    filelist=[]
    for file in sorted(os.listdir(args.pymaf_path)):
        if 'param.npy' in file:
            filelist.append(os.path.join(args.pymaf_path, file))

    filelist2=[]
    for file in sorted(os.listdir(args.pixie_path)):
        if 'param.pkl' in file:
            filelist2.append(os.path.join(args.pixie_path, file))

    # print(len(filelist), len(filelist2))
    assert len(filelist)==len(filelist2)
    
    pose_dict = {}
    pose_dict['full_pose'] = torch.empty((1,len(filelist),55,3,3))
    pose_dict['exp'] = torch.zeros((1,len(filelist),10)).cuda()
    pose_dict['cam'] = torch.empty((1,len(filelist),3)).cuda()

    jaw_pose = torch.eye(3, dtype=torch.float32).unsqueeze(0) #param_dict['jaw_pose']
    eye_pose = torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(2,1,1)
    global_pose = torch.FloatTensor([[[ 1, -0.0000, -0.00],
                                 [-0.00, -1, -0],
                                 [-0.0,  0.00, -1]]])

    for i,filepath in enumerate(filelist):

        with open(filelist2[i], 'rb') as f:
            codedict = pickle.load(f)
            param_dict = {}
            for key in codedict.keys():
                if isinstance(codedict[key], str):
                    param_dict[key] = codedict[key]
                else:
                    param_dict[key] = torch.from_numpy(codedict[key])

            # pixie_pose = torch.cat([param_dict['global_pose'], param_dict['body_pose'],
            #                     jaw_pose, eye_pose,
            #                     param_dict['left_hand_pose'], param_dict['right_hand_pose']], dim=0)

            pixie_pose = torch.cat([global_pose, param_dict['body_pose'],
                                jaw_pose, eye_pose,
                                param_dict['left_hand_pose'], param_dict['right_hand_pose']], dim=0)

            # print(param_dict['global_pose'])

        pymaf_pose = torch.from_numpy(np.load(filepath))

        # hybrid_pose = 

        # pose_dict['full_pose'][0][i] = torch.from_numpy(np.load(filepath))
        pose_dict['full_pose'][0][i] = pixie_pose
        # pose_dict['cam'][0][i] = param_dict['body_cam'].squeeze()
        pose_dict['cam'][0][i] = torch.Tensor([1.424,0.0035,0.453]) #upper body?
        # pose_dict['cam'][0][i] = torch.Tensor([0.7,0.00,0.0])

    os.makedirs(args.save_path, exist_ok=True)
    util.save_pkl(os.path.join(args.save_path, 'pose_hybrid.pkl'), pose_dict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='generate dataset from video or image folder')
    parser.add_argument('--pymaf_path', default='../poses/id2_pose3/pymaf/frame', type=str,
                        help='')
    parser.add_argument('--pixie_path', default='../poses/id2_pose3/pixie', type=str,
                        help='')
    parser.add_argument("--save_path", default='', type=str,
                        help = '')
    args = parser.parse_args()

    main(args)

