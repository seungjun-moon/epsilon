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

def pose_smooth(pose, weight=[0.1, 0.25, 0.5, 0.75, 0.9], mi=0): #mi = mixing index
    weight_o =weight.copy()
    weight.reverse()
    new_weight=weight_o+[1]+weight

    smooth_pose = torch.empty(pose.shape) # 300 * 55 * 3 * 3
    interval = len(weight)
    total_pose = torch.empty((pose.shape[0]+2*interval,*pose.shape[1:]))
    for i in range(total_pose.shape[0]):
        if i < interval:
            total_pose[i] = pose[0]
        elif i >= total_pose.shape[0]-interval:
            total_pose[i] = pose[-1]
        else:
            total_pose[i] = pose[i-interval]

    for i in range(smooth_pose.shape[0]):
        smooth_pose[i][:mi] = total_pose[i+interval][:mi]
        for j in range(len(new_weight)):
            if j==0:
                smooth_pose[i][mi:] = (new_weight[j] * total_pose[i+j][mi:]) / sum(new_weight)
            else:
                smooth_pose[i][mi:] += (new_weight[j] * total_pose[i+j][mi:]) / sum(new_weight)

    return smooth_pose


def generate_frame(inputpath, savepath, subject_name=None, n_frames=2000, fps=30):
    ''' extract frames from video or copy frames from image folder
    '''
    os.makedirs(savepath, exist_ok=True)
    if subject_name is None:
        subject_name = Path(inputpath).stem
    ## video data
    if os.path.isfile(inputpath) and (os.path.splitext(inputpath)[-1] in ['.mp4', '.csv', '.MOV']):
        videopath = os.path.join(os.path.dirname(savepath), f'{subject_name}.mp4')
        logger.info(f'extract frames from video: {inputpath}..., then save to {videopath}')        
        vidcap = cv2.VideoCapture(inputpath)
        count = 0
        success, image = vidcap.read()
        cv2.imwrite(os.path.join(savepath, f'{subject_name}_f{count:06d}.png'), image) 
        h, w = image.shape[:2]
        # import imageio
        # savecap = imageio.get_writer(videopath, fps=fps)
        # savecap.append_data(image[:,:,::-1])
        while success:
            count += 1
            success,image = vidcap.read()
            if count > n_frames or image is None:
                break
            imagepath = os.path.join(savepath, f'{subject_name}_f{count:06d}.png')
            cv2.imwrite(imagepath, image)     # save frame as JPEG png
            # savecap.append_data(image[:,:,::-1])
        logger.info(f'extracted {count} frames')
    elif os.path.isdir(inputpath):
        logger.info(f'copy frames from folder: {inputpath}...')
        imagepath_list = glob(inputpath + '/*.jpg') +  glob(inputpath + '/*.png') + glob(inputpath + '/*.jpeg')
        imagepath_list = sorted(imagepath_list)
        for count, imagepath in enumerate(imagepath_list):
            shutil.copyfile(imagepath, os.path.join(savepath, f'{subject_name}_f{count:06d}.png'))
        print('frames are stored in {}'.format(savepath))
    else:
        logger.info(f'please check the input path: {inputpath}')
    logger.info(f'video frames are stored in {savepath}')

def generate_image(inputpath, savepath, subject_name=None, crop=False, image_size=512, scale_bbox=1.1, device='cuda:0'):
    ''' generate image from given frame path. 
    '''
    logger.info(f'generae images, crop {crop}, image size {image_size}')
    os.makedirs(savepath, exist_ok=True)
    # load detection model
    from submodules.detector import FasterRCNN
    detector = FasterRCNN(device=device)
    if os.path.isdir(inputpath):
        imagepath_list = glob(inputpath + '/*.jpg') +  glob(inputpath + '/*.png') + glob(inputpath + '/*.jpeg')
        imagepath_list = sorted(imagepath_list)
        # if crop, detect the bbox of the first image and use the bbox for all frames
        if crop:
            imagepath = imagepath_list[0]
            logger.info(f'detect first image {imagepath}')
            imagename = os.path.splitext(os.path.basename(imagepath))[0]
            image = imread(imagepath)[:,:,:3]/255.
            h, w, _ = image.shape

            image_tensor = torch.tensor(image.transpose(2,0,1), dtype=torch.float32)[None, ...]
            bbox = detector.run(image_tensor)
            left = bbox[0]; right = bbox[2]; top = bbox[1]; bottom = bbox[3]
            np.savetxt(os.path.join(Path(inputpath).parent, 'image_bbox.txt'), bbox)
            
            ## calculate warping function for image cropping
            old_size = max(right - left, bottom - top)
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
            size = int(old_size*scale_bbox)
            src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
            DST_PTS = np.array([[0,0], [0,image_size - 1], [image_size - 1, 0]])
            tform = estimate_transform('similarity', src_pts, DST_PTS)
        
        for count, imagepath in enumerate(tqdm(imagepath_list)):
            if crop:
                image = imread(imagepath)
                dst_image = warp(image, tform.inverse, output_shape=(image_size, image_size))
                dst_image = (dst_image*255).astype(np.uint8)
                imsave(os.path.join(savepath, f'{subject_name}_f{count:06d}.png'), dst_image)
            else:
                shutil.copyfile(imagepath, os.path.join(savepath, f'{subject_name}_f{count:06d}.png'))
    logger.info(f'images are stored in {savepath}')

def generate_pixie(inputpath, savepath, ckpt_path='assets/face_normals/model.pth', device='cuda:0', image_size=512, vis=False):
    logger.info(f'generate pixie results')
    os.makedirs(savepath, exist_ok=True)
    # load model
    sys.path.insert(0, './submodules/PIXIE')
    from pixielib.pixie import PIXIE
    from pixielib.datasets.body_datasets import TestData
    from pixielib.utils import util
    from pixielib.visualizer import Visualizer
    from pixielib.utils.config import cfg as pixie_cfg
    from pixielib.utils.tensor_cropper import transform_points
    # run pixie
    testdata = TestData(inputpath, iscrop=False)
    pixie_cfg.model.use_tex = False
    pixie_cfg.model.n_exp = 10 #Default : 50, but I changed for the reenactment.
    pixie = PIXIE(config = pixie_cfg, device=device)
    visualizer = Visualizer(render_size=image_size, config = pixie_cfg, device=device, rasterizer_type='standard')
    testdata = TestData(inputpath, iscrop=False)

    pose_dict = {}
    pose_dict['full_pose'] = torch.empty((1,len(testdata),55,3,3)).cuda()
    pose_dict['exp'] = torch.empty((1,len(testdata),10)).cuda()
    pose_dict['cam'] = torch.empty((1,len(testdata),3)).cuda()


    key = 'body'
    for i, batch in enumerate(tqdm(testdata, dynamic_ncols=True)):
        batch['image'] = batch['image'].unsqueeze(0)
        batch['image_hd'] = batch['image_hd'].unsqueeze(0)
        name = batch['name']
    
        util.move_dict_to_device(batch, device)
        data = {
            key : batch
        }

        param_dict = pixie.encode(data, threthold=True, keep_local=True, copy_and_paste=False)
        codedict = param_dict[key]

        # if i == 0:
        #     print(codedict['head_pose']) #non-deterministic

        ## Visualize Intermediate Outputs.

        # image_head = torch.squeeze(data['body']['image_hd'], dim=0).cpu().numpy()*255
        # image_head = np.transpose(image_head, (2,1,0))
        # image_head = Image.fromarray(image_head.astype("uint8"), mode="RGB")
        # image_head.save(os.path.join(savepath, f'{name}_image_hd.jpg'))

        # image_head = torch.squeeze(data['body']['head_image'], dim=0).cpu().numpy()*255
        # image_head = np.transpose(image_head, (2,1,0))
        # image_head = Image.fromarray(image_head.astype("uint8"), mode="RGB")
        # image_head.save(os.path.join(savepath, f'{name}_head.jpg'))

        # image_hand1 = torch.squeeze(data['body']['left_hand_image'], dim=0).cpu().numpy()*255
        # image_hand1 = np.transpose(image_hand1, (2,1,0))
        # image_hand1 = Image.fromarray(image_hand1.astype("uint8"), mode="RGB")
        # image_hand1.save(os.path.join(savepath, f'{name}_hand1.jpg'))

        # image_hand2 = torch.squeeze(data['body']['right_hand_image'], dim=0).cpu().numpy()*255
        # image_hand2 = np.transpose(image_hand2, (2,1,0))
        # image_hand2 = Image.fromarray(image_hand2.astype("uint8"), mode="RGB")
        # image_hand2.save(os.path.join(savepath, f'{name}_hand2.jpg'))

        ###################################

        opdict = pixie.decode(codedict, param_type=key) #originally, body --> Strange
        if vis:
            opdict['albedo'] = visualizer.tex_flame2smplx(opdict['albedo'])
            visdict = visualizer.render_results(opdict, data['body']['image_hd'], overlay=True, use_deca=False)
            cv2.imwrite(os.path.join(savepath, f'{name}_vis.jpg'), visualizer.visualize_grid(visdict, size=image_size))

        full_pose = torch.cat([codedict['global_pose'][0], codedict['body_pose'][0], codedict['jaw_pose'][0], \
                              torch.eye(3, dtype=codedict['body_pose'].dtype, device=codedict['body_pose'].device).unsqueeze(0).repeat(2,1,1), \
                              codedict['left_hand_pose'][0], codedict['right_hand_pose'][0]], dim=0)

        # cam = codedict['body_cam']
        # cam = torch.Tensor([0.7, 0., 0.]).cuda()
        cam = torch.Tensor([1.424,0.0035,0.453])
        exp = codedict['exp']

        pose_dict['full_pose'][0][i] = full_pose
        pose_dict['cam'][0][i] = cam
        pose_dict['exp'][0][i] = exp

    # util.save_pkl(os.path.join(savepath, 'pose.pkl'), pose_dict)

    return pose_dict, pose_dict['full_pose']


def extract_pose(subjectpath, savepath=None, vis=False, crop=False, ignore_existing=False, n_frames=2000):
    if savepath is None:
        savepath = Path(subjectpath).parent
    subject_name = Path(subjectpath).stem
    savepath = os.path.join(savepath, subject_name)
    os.makedirs(savepath, exist_ok=True)
    logger.info(f'processing {subject_name}')
    # 0. copy frames from video or image folder
    if ignore_existing or not os.path.exists(os.path.join(savepath, 'frame')):
        generate_frame(subjectpath, os.path.join(savepath, 'frame'), n_frames=n_frames)
    # 1. crop image from frames, use fasterrcnn for detection
    if ignore_existing or not os.path.exists(os.path.join(savepath, 'image')):
        generate_image(os.path.join(savepath, 'frame'), os.path.join(savepath, 'image'), subject_name=subject_name,
                        crop=crop, image_size=512, scale_bbox=1.1, device='cuda:0')
    # 4. smplx estimation using PIXIE (https://github.com/yfeng95/PIXIE)
    if ignore_existing or not os.path.exists(os.path.join(savepath, 'pixie')):
        pose_dict, full_pose = generate_pixie(os.path.join(savepath, 'image'), os.path.join(savepath, 'pixie'), vis=vis)

    return pose_dict, full_pose

    # logger.info(f'finish {subject_name}')

def main(args):
    
    with open(args.list, 'r') as f:
        lines = f.readlines()
    subject_list = [s.strip() for s in lines]

    for subjectpath in tqdm(subject_list):
        full_pose_list=[]
        for i in range(args.smooth):
            pose_dict, full_pose = extract_pose(subjectpath, savepath=args.savepath, vis=args.vis, crop=args.crop, ignore_existing=args.ignore_existing, n_frames=args.n_frames)
            full_pose_list.append(full_pose)
        
        for i in range(args.smooth):
            if i==0:
                full_pose = full_pose_list[i]
            else:
                full_pose +=full_pose_list[i]
        full_pose = full_pose/args.smooth
        
        smooth_pose = pose_smooth(full_pose)

        pose_dict['full_pose'] = smooth_pose

        ##### fuck no ####
        # import pickle
        # pose_dict = pickle.load(open('/home/june1212/inscarf/poses/id2_pose3/pixie/pose_rough.pkl', 'rb'))

        # smooth_pose = pose_smooth(torch.from_numpy(pose_dict['full_pose']))

        # pose_dict['full_pose'] = smooth_pose.numpy()
        # sys.path.insert(0, './submodules/PIXIE')

        ##### fuck no ####

        from pixielib.utils import util
        savepath = os.path.join(Path(subjectpath).parent, Path(subjectpath).stem, 'pixie')
        print(savepath)
        util.save_pkl(os.path.join(savepath, 'pose.pkl'), pose_dict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='generate dataset from video or image folder')
    parser.add_argument('--list', default='lists/pose_video_list.txt', type=str,
                        help='path to the subject data, can be image folder or video')
    parser.add_argument('--savepath', default=None, type=str,
                        help='path to save processed data, if not specified, then save to the same folder as the subject data')
    parser.add_argument("--image_size", default=512, type=int,
                        help = 'image size')
    parser.add_argument("--crop", default=True, action="store_true",
                        help='whether to crop image according to the subject detection bbox')
    parser.add_argument("--vis", default=True, action="store_true",
                        help='whether to visualize labels (lmk, iris, face parsing)')
    parser.add_argument("--ignore_existing", default=False, action="store_true",
                        help='ignore existing data')
    parser.add_argument("--n_frames", default=800, type=int,
                        help='number of frames to be processed')
    parser.add_argument("--smooth", default=1, type=int,
                        help='ensemble model number')
    args = parser.parse_args()

    main(args)

