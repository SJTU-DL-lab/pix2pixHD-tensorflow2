import tensorflow as tf
from tensorflow.python.tools import freeze_graph

input_binary=True
input_graph='/home/yaosy/Diskb/projects/pix2pixHD-tensorflow2/graph.pb'
input_checkpoint='/home/yaosy/Diskb/projects/pix2pixHD-tensorflow2/checkpoints/debug_train_graph/train_ckpt/ckpt-25'
output_node_names="discriminator/"
output_graph='/home/yaosy/Diskb/projects/pix2pixHD-tensorflow2/freeze_model.pb'

freeze_graph.freeze_graph(input_graph,
             input_binary=input_binary,
             input_saver='',
             input_checkpoint=input_checkpoint,
             output_node_names=output_node_names,
             output_graph=output_graph,
             restore_op_name='',
             filename_tensor_name='',
             clear_devices=True,
             initializer_nodes='')
