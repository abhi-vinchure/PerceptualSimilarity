TRIAL=${1}
NET=${2}
mkdir checkpoints

mkdir checkpoints/${NET}_${TRIAL}
python3 ./train.py --use_gpu --net ${NET} --name ${NET}_${TRIAL}
python3 ./test_dataset_model.py --use_gpu --net ${NET} --model_path ./checkpoints/${NET}_${TRIAL}/latest_net_.pth

mkdir checkpoints/${NET}_${TRIAL}_scratch
python3 ./train.py --from_scratch --train_trunk --use_gpu --net ${NET} --name ${NET}_${TRIAL}_scratch
python3 ./test_dataset_model.py --from_scratch --train_trunk --use_gpu --net ${NET} --model_path ./checkpoints/${NET}_${TRIAL}_scratch/latest_net_.pth

mkdir checkpoints/${NET}_${TRIAL}_tune
python3 ./train.py --train_trunk --use_gpu --net ${NET} --name ${NET}_${TRIAL}_tune
python3 ./test_dataset_model.py --train_trunk --use_gpu --net ${NET} --model_path ./checkpoints/${NET}_${TRIAL}_tune/latest_net_.pth
