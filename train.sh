#!/bin/bash

# train
# set -v
# i=0
# for hidden in 400 600 800 1000;do
#   for bottle_num in 500 700 900 1100 1300;do
#     i=`expr $i + 1`
#     python3 train.py -s RESULTS/test1018_${i} -t Amide -td all -e 0 0 1000 --be_num 5 --disc_weight 1 --cls_order_weight 1 0.05 --label_smooth 0.0 --schedual_stones 2000 --data_normalization minmax --ae_disc_lr 0.0002 0.01 -bs 60 --ae_units 2000 2000 --disc_units $hidden $hidden --net_type simple --early_stop --resnet_bottle_num 50 50 --bottle_num $bottle_num --denoise 0.0 --optim adam --order_losstype paired_ce --reconst_loss ce --disc_weight_epoch 500 --early_stop_check_num 100 --use_batch_for_order --dropouts 0.0 0.0 0.0 0.0
#   done
# done
# i=20
# for bs in 30 60 90 120;do
#   i=`expr $i + 1`
#   python3 train.py -s RESULTS/test1018_${i} -t Amide -td all -e 0 0 1000 --be_num 5 --disc_weight 1 --cls_order_weight 1 0.05 --label_smooth 0.0 --schedual_stones 2000 --data_normalization minmax --ae_disc_lr 0.0002 0.01 -bs $bs --ae_units 2000 2000 --disc_units 1000 1000 --net_type simple --early_stop --resnet_bottle_num 50 50 --bottle_num 500 --denoise 0.0 --optim adam --order_losstype paired_ce --reconst_loss ce --disc_weight_epoch 500 --early_stop_check_num 100 --use_batch_for_order --dropouts 0.0 0.0 0.0 0.0
# done
# for((i=25;i<=28;i++));do
#   j=`expr $i - 1`
#   python3 train.py -s RESULTS/test1018_${i} -t Amide -td all -e 0 0 1000 --be_num 5 --disc_weight 1 --cls_order_weight 1 0.05 --label_smooth 0.0 --schedual_stones 2000 --data_normalization minmax --ae_disc_lr 0.0002 0.01 -bs 60 --ae_units 2000 2000 --disc_units 1000 1000 --net_type simple --early_stop --resnet_bottle_num 50 50 --bottle_num 500 --denoise 0.0 --optim adam --order_losstype paired_ce --reconst_loss ce --disc_weight_epoch 500 --early_stop_check_num 100 --use_batch_for_order --dropouts 0.0 0.0 0.0 0.0 --load_model RESULTS/test1018_${j}/models.pth
# done

# generate
# set -v
# for((i=25;i<=28;i++));do
#   python3 generate.py RESULTS/test1018_${i}
# done

# evaluation
# set -v
for((i=25;i<=28;i++));do
  python3 evaluation_ml.py RESULTS/test1018_${i}
done
