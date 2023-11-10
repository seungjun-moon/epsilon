import os, sys
import argparse
import shutil
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lib.trainer import Trainer
    
def train(subject_name, exp_cfg, args=None):
    cfg = get_cfg_defaults()
    cfg = update_cfg(cfg, exp_cfg)
    cfg = update_cfg(cfg, data_cfg)
    cfg.cfg_file = data_cfg
    cfg.group = data_type
    cfg.dataset.path = os.path.abspath(cfg.dataset.path)
    cfg.clean = args.clean
    cfg.output_dir = os.path.join(args.exp_dir, data_type, subject_name)
    cfg.wandb_name = args.wandb_name
    
    cfg.ero = ero
    cfg.eio = eio

    if 'nerf' in exp_cfg:
        cfg.exp_name = f'{subject_name}_nerf'
        cfg.output_dir = os.path.join(args.exp_dir, data_type, subject_name, args.dir_stage_0)
        # cfg.ckpt_path = os.path.abspath('./exps/data/sj_klleon2/nerf/model.tar') # any pretrained nerf model to have a better initialization
        cfg.ckpt_path = args.ckpt_path
    else:
        cfg.exp_name = f'{subject_name}_hybrid'
        cfg.output_dir = os.path.join(args.exp_dir, data_type, subject_name, args.dir_stage_1)
        cfg.ckpt_path = os.path.join(args.exp_dir, data_type, subject_name, args.dir_stage_0, 'model.tar')
    if args.clean:
        shutil.rmtree(os.path.join(cfg.output_dir, f'{cfg.group}/{cfg.exp_name}'), ignore_errors=True)
    os.makedirs(os.path.join(cfg.output_dir), exist_ok=True)
    shutil.copy(data_cfg, os.path.join(cfg.output_dir, 'config.yaml'))
    shutil.copy(exp_cfg, os.path.join(cfg.output_dir, 'exp_config.yaml'))
    # creat folders 
    os.makedirs(os.path.join(cfg.output_dir, cfg.train.log_dir), exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, cfg.train.vis_dir), exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, cfg.train.val_vis_dir), exist_ok=True)
    with open(os.path.join(cfg.output_dir, cfg.train.log_dir, 'full_config.yaml'), 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)
    # start training
    trainer = Trainer(config=cfg)
    trainer.fit()

if __name__ == '__main__':
    from lib.utils.config import get_cfg_defaults, update_cfg
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb_name', type=str, default = 'EPSilon', help='project name')
    parser.add_argument('--exp_dir', type=str, default = './exps', help='exp dir')
    parser.add_argument('--dir_stage_0', type=str, default = 'nerf', help='directory name for stage 0')
    parser.add_argument('--dir_stage_1', type=str, default = 'hybrid', help='directory name for stage 1')
    parser.add_argument('--data_cfg', type=str, default = 'configs/data/mpiis/DSC_7157.yml', help='data cfg file path')
    parser.add_argument('--train_nerf', action="store_true", help='')
    parser.add_argument('--clean', action="store_true", help='delete output dir if exists')
    parser.add_argument('--ckpt_path', type=str, default='./exps/data/sj_klleon2/nerf/model.tar', help='')

    parser.add_argument('--ero', action="store_true", help='apply ERO')
    parser.add_argument('--eio', action="store_true", help='apply EIO')
    args = parser.parse_args()
    # 
    #-- data setting
    data_cfg = args.data_cfg
    data_type = data_cfg.split('/')[-2]
    subject_name = data_cfg.split('/')[-1].split('.')[0]
    
    ### ------------- start training
    if args.train_nerf:
        exp_cfg = 'configs/exp/stage_0_nerf.yml'
        train(subject_name, exp_cfg, args)
    exp_cfg = 'configs/exp/stage_1_hybrid.yml'
    train(subject_name, exp_cfg, args)



