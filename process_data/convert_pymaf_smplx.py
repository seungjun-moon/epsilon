import os, sys
import argparse
import torch
import numpy as np

sys.path.insert(0, './submodules/PIXIE')
from pixielib.utils import util

def main(args):
    filelist=[]
    for file in sorted(os.listdir(args.pymaf_path)):
        if 'param.npy' in file:
            filelist.append(os.path.join(args.pymaf_path, file))
    
    pose_dict = {}
    pose_dict['full_pose'] = torch.empty((1,len(filelist),55,3,3))
    pose_dict['exp'] = torch.zeros((1,len(filelist),10)).cuda()
    pose_dict['cam'] = torch.empty((1,len(filelist),3)).cuda()

    for i,filepath in enumerate(filelist):
        pose_dict['full_pose'][0][i] = torch.from_numpy(np.load(filepath))
        pose_dict['cam'][0][i] = torch.Tensor([1.424,0.0035,0.453]) #upper body?
        # pose_dict['cam'][0][i] = torch.Tensor([0.7,0.00,0.0])
    
    util.save_pkl(os.path.join(args.save_path, 'pose_pymaf.pkl'), pose_dict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='generate dataset from video or image folder')
    parser.add_argument('--pymaf_path', default='lists/pose_video_list.txt', type=str,
                        help='')
    parser.add_argument("--save_path", default='', type=str,
                        help = '')
    args = parser.parse_args()

    main(args)

