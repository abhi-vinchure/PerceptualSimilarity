
TRIAL=${1}
NET=${2}
mkdir checkpoints
mkdir checkpoints/${NET}_${TRIAL}_tune
python3 ./train.py --train_trunk --use_gpu --net ${NET} --name ${NET}_${TRIAL}_tune
python3 ./test_dataset_model.py --train_trunk --use_gpu --net ${NET} --model_path ./checkpoints/${NET}_${TRIAL}_tune/latest_net_.pth
