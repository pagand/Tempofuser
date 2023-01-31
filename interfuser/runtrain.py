import os
GPU_NUM=1
DATASET_ROOT='../dataset/'
import train0

if __name__ == "__main__":
    train0.main()


# os.system("python3 -m torch.distributed.launch --nproc_per_node=1 train0.py ../dataset/ --dataset carla --train-towns 3  --val-towns 7 \
#                --train-weathers 0 1 2  --val-weathers 2 \
#                --model interfuser_baseline --sched cosine --epochs 25 --warmup-epochs 5 --lr 0.0005 --batch-size 16  -j 1 --no-prefetcher --eval-metric l1_error \
#                --opt adamw --opt-eps 1e-8 --weight-decay 0.05  \
#                --scale 0.9 1.1 --saver-decreasing --clip-grad 10 --freeze-num -1 \
#                --with-backbone-lr --backbone-lr 0.0002 \
#                --multi-view --with-lidar --multi-view-input-size 3 128 128 \
#                --experiment interfuser_baseline \
#                --pretrained")