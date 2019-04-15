python /home/yaosy/anaconda3/envs/tf2/lib/python3.6/site-packages/tensorflow/python/tools/freeze_graph.py \
--input_binary=true \
--input_graph /home/yaosy/Diskb/projects/pix2pixHD-tensorflow2/graph.pb \
--input_checkpoint /home/yaosy/Diskb/projects/pix2pixHD-tensorflow2/checkpoints/debug_train_graph/train_ckpt/ckpt-1 \
--output_node_names global_generator/sequential/tanh/Tanh \
--output_graph /home/yaosy/Diskb/projects/pix2pixHD-tensorflow2/freeze_model.pb
