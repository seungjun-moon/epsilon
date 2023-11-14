## Preprocess

CUDA_VISIBLE_DEVICES=3 python main_train.py --data_cfg ./configs/data/snapshot/male-3-casual.yml \
--exp_dir exps --dir_stage_1 hybrid --ckpt_path ../epsilon_old/exps_snapshot/snapshot/male-3-casual/nerf/model.tar \
--ero --eio

python main_demo.py --vis_type capture --frame_id 0 --model_path exps/snapshot/male-3-casual/hybrid

CUDA_VISIBLE_DEVICES=0 python main_demo.py --vis_type animate \
--model_path exps/snapshot/male-3-casual/hybrid --animation_file 'data/pixie_radioactive.pkl'