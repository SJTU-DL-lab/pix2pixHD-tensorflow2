python /home/yaosy/anaconda3/envs/tf2/lib/python3.6/site-packages/tensorflow/python/tools/inspect_checkpoint.py \
--file_name /home/yaosy/Diskb/projects/pix2pixHD-tensorflow2/checkpoints/debug_train_graph/train_ckpt/ckpt-25 \
--all_tensors

# python /home/yaosy/anaconda3/envs/tf2/lib/python3.6/site-packages/tensorflow/python/tools/freeze_graph.py \
# --input_binary=False \
# --input_graph /home/yaosy/Diskb/projects/pix2pixHD-tensorflow2/graph.pbtxt \
# --input_checkpoint /home/yaosy/Diskb/projects/pix2pixHD-tensorflow2/checkpoints/debug_train_graph/train_ckpt/ckpt-4 \
# --output_node_names global_generator \
# --output_graph /home/yaosy/Diskb/projects/pix2pixHD-tensorflow2/freeze_model.pb
