## Preprocess

CUDA_VISIBLE_DEVICES=3 python main_train.py --data_cfg ./configs/data/snapshot/male-3-casual.yml \
--exp_dir exps --dir_stage_1 hybrid --ckpt_path ../epsilon_old/exps_snapshot/snapshot/male-3-casual/nerf/model.tar \
--ero --eio

CUDA_VISIBLE_DEVICES=0 python main_demo.py --vis_type capture --model_path exps_legacy