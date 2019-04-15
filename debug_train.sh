python train.py --label_nc 18 \
--name debug_train_graph \
--dataroot /home/yaosy/Document/projects/deep-imitation-train/data/target/gray_maskgirl \
--resize_or_crop resize --norm instance --no_instance \
--loadSize 32 --batchSize 8 --save_epoch_freq 1 \
--display_freq 10 --no_shuffle --no_flip
