from ipaddress import ip_address
import os, sys
import argparse
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from glob import glob
from skimage.io import imread, imsave
from skimage.transform import estimate_transform, warp, resize, rescale
import shutil
import torch
import cv2
import numpy as np
from PIL import Image

def generate_pixie(inputpath, savepath, ckpt_path='assets/face_normals/model.pth', device='cuda:0', image_size=512, vis=False):
    logger.info(f'generate pixie results')
    os.makedirs(savepath, exist_ok=True)
    # load model
    sys.path.insert(0, './submodules/PIXIE')
    from pixielib.pixie import PIXIE
    from pixielib.visualizer import Visualizer
    from pixielib.datasets.body_datasets import TestData
    from pixielib.utils import util
    from pixielib.utils.config import cfg as pixie_cfg
    from pixielib.utils.tensor_cropper import transform_points
    # run pixie
    testdata = TestData(inputpath, iscrop=False)
    pixie_cfg.model.use_tex = False
    pixie = PIXIE(config = pixie_cfg, device=device)
    visualizer = Visualizer(render_size=image_size, config = pixie_cfg, device=device, rasterizer_type='standard')
    testdata = TestData(inputpath, iscrop=False)
    for i, batch in enumerate(tqdm(testdata, dynamic_ncols=True)):
        batch['image'] = batch['image'].unsqueeze(0)
        batch['image_hd'] = batch['image_hd'].unsqueeze(0)
        name = batch['name']
        util.move_dict_to_device(batch, device)
        data = {
            'body': batch
        }
        param_dict = pixie.encode(data, threthold=True, keep_local=True, copy_and_paste=False)
        codedict = param_dict['body']
        opdict = pixie.decode(codedict, param_type='body')
        util.save_pkl(os.path.join(savepath, f'{name}_param.pkl'), codedict)
        if vis:
            opdict['albedo'] = visualizer.tex_flame2smplx(opdict['albedo'])
            visdict = visualizer.render_results(opdict, data['body']['image_hd'], overlay=True, use_deca=False)
            cv2.imwrite(os.path.join(savepath, f'{name}_vis.jpg'), visualizer.visualize_grid(visdict, size=image_size))

def main(args):
    generate_pixie(os.path.join(args.image_folder), os.path.join(args.output_folder), vis=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='generate dataset from video or image folder')
    parser.add_argument('--image_folder', type=str)
    parser.add_argument('--output_folder', type=str)
    args = parser.parse_args()

    main(args)


