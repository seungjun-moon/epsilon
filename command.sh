## Preprocess

CUDA_VISIBLE_DEVICES=3 python main_train.py --data_cfg ./configs/data/snapshot/male-3-casual.yml \
--exp_dir exps_dummy --dir_stage_1 hybrid --ckpt_path '../scarf/exps/mpiis/DSC_7157/model.tar' \
--ero --eio

CUDA_VISIBLE_DEVICES=0 python main_demo.py --vis_type capture --model_path exps_legacy