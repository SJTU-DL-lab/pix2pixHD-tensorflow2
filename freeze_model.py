import tensorflow as tf
from tensorflow.python.tools import freeze_graph

input_binary=False
input_graph='./checkpoints/debug_train_graph/net_G_savedModel/saved_model.pb'
input_checkpoint='checkpoints/debug_train_graph/net_G_savedModel/variables/variables'
output_node_names="output_1"
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
