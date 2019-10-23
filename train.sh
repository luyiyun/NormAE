#!/bin/bash

# train
# set -v
# i=1
# for rec_train in 0 100 300 500 700 900 1000;do
#   for hiddens in 1000 2000 3000 4000 5000 6000;do
#     i=`expr $i + 1`
#     python3 train.py -s RESULTS/test1020_${i} -t Amide -td all -e ${rec_train} 100 400 --be_num 5 --disc_weight 1 --cls_order_weight 0.5 0.05 --label_smooth 0.0 --schedual_stones 2000 --data_normalization standard --ae_disc_lr 0.0002 0.01 -bs 60 --ae_units ${hiddens} --disc_units 1000 1000 --net_type simple --early_stop --resnet_bottle_num 50 50 --bottle_num 500 --denoise 0.0 --optim adam --order_losstype paired_ce --reconst_loss mae --disc_weight_epoch 100 --early_stop_check_num 100 --use_batch_for_order --dropouts 0.0 0.0 0.0 0.0 --visdom_env test1020_${i}
#   done
# done
# for bs in 30 60 90 120;do
#   i=`expr $i + 1`
#   python3 train.py -s RESULTS/test1018_${i} -t Amide -td all -e 0 0 1000 --be_num 5 --disc_weight 1 --cls_order_weight 1 0.05 --label_smooth 0.0 --schedual_stones 2000 --data_normalization minmax --ae_disc_lr 0.0002 0.01 -bs $bs --ae_units 2000 2000 --disc_units 1000 1000 --net_type simple --early_stop --resnet_bottle_num 50 50 --bottle_num 500 --denoise 0.0 --optim adam --order_losstype paired_ce --reconst_loss ce --disc_weight_epoch 500 --early_stop_check_num 100 --use_batch_for_order --dropouts 0.0 0.0 0.0 0.0
# done
# i=1
# for rec_train in 0 100 300 500 700 900 1000;do
#   for hiddens in 1000 2000 3000 4000 5000 6000;do
#     i=`expr $i + 1`
#     python3 train.py -s RESULTS/test1021_${i} -t Amide -td all -e 0 0 100 --be_num 5 --disc_weight 1 --cls_order_weight 0.5 0.05 --label_smooth 0.0 --schedual_stones 2000 --data_normalization standard --ae_disc_lr 0.0002 0.01 -bs 60 --ae_units ${hiddens} --disc_units 1000 1000 --net_type simple --early_stop --resnet_bottle_num 50 50 --bottle_num 500 --denoise 0.0 --optim adam --order_losstype paired_ce --reconst_loss mae --disc_weight_epoch 100 --early_stop_check_num 100 --use_batch_for_order --dropouts 0.0 0.0 0.0 0.0 --load_model RESULTS/test1020_${i}/models.pth --visdom_env test1021_${i}
#   done
# done
# i=0
# for disc_train in 0 100 200 300 400;do
#   for disc_hiddens in 100 300 500 700 1000;do
#     i=`expr $i + 1`
#     python3 train.py -s RESULTS/test1022_${i} -t Amide -td all \
#       -e 500 ${disc_train} 700 --be_num 5 --disc_weight 1 \
#       --cls_order_weight 0.5 0.05 --label_smooth 0.0 --schedual_stones 2000 \
#       --data_normalization standard --ae_disc_lr 0.0002 0.01 -bs 60 \
#       --ae_units 3000 --disc_units ${disc_hiddens} ${disc_hiddens} \
#       --net_type simple --early_stop --resnet_bottle_num 50 50 \
#       --bottle_num 500 --denoise 0.0 --optim adam --order_losstype paired_ce \
#       --reconst_loss mae --disc_weight_epoch 100 --early_stop_check_num 200 \
#       --use_batch_for_order --dropouts 0.0 0.0 0.0 0.0 \
#       --visdom_env test1022_${i}
#   done
# done
# i=0
# for ae_unit in 1000 1500 2000 2500 3000;do
#   for code_num in 200 400 600 800 1000;do
#     i=`expr $i + 1`
#     python3 train.py -s RESULTS/test1023_${i} -t Amide -td all \
#       -e 500 100 1000 --be_num 5 --disc_weight 1 \
#       --cls_order_weight 0.5 0.05 --label_smooth 0.0 --schedual_stones 2000 \
#       --data_normalization standard --ae_disc_lr 0.0002 0.01 -bs 60 \
#       --ae_units ${ae_unit} ${ae_unit} --disc_units 500 500 \
#       --net_type simple --early_stop --resnet_bottle_num 50 50 \
#       --bottle_num ${code_num} --denoise 0.0 --optim adam --order_losstype paired_ce \
#       --reconst_loss mae --disc_weight_epoch 100 --early_stop_check_num 200 \
#       --use_batch_for_order --dropouts 0.0 0.0 0.0 0.0 \
#       --visdom_env test1023_${i}
#   done
# done

# generate
# set -v
# for((i=1;i<=25;i++));do
#   python3 generate.py RESULTS/test1022_${i}
# done

# evaluation
# set -v
for((i=1;i<=25;i++));do
  python3 evaluation_ml.py RESULTS/test1023_${i}
done

