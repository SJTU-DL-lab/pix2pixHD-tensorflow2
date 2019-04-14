python train.py --label_nc 18 \
--name debug_train \
--dataroot /home/yaosy/Document/projects/deep-imitation-train/data/target/gray_maskgirl \
--resize_or_crop resize --norm instance --no_instance \
--loadSize 128 --batchSize 1 --save_epoch_freq 20 \
--display_freq 20 --no_shuffle --no_flip
