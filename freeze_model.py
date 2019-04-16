import tensorflow as tf
from tensorflow.python.tools import freeze_graph

# input_binary=False
# input_graph='./checkpoints/debug_train_graph/net_G_savedModel/saved_model.pb'
# input_checkpoint='checkpoints/debug_train_graph/net_G_savedModel/variables/variables'
# output_node_names="output_1"
# output_graph='/home/yaosy/Diskb/projects/pix2pixHD-tensorflow2/freeze_model.pb'

saved_model_dir = './checkpoints/debug_train_graph/net_G_savedModel'
output_graph_filename = "./output_graph.pb"

input_saved_model_dir = saved_model_dir
output_node_names = "global_generator/sequential/tanh/Tanh"
input_binary = False
input_saver_def_path = False
restore_op_name = None
filename_tensor_name = None
clear_devices = False
input_meta_graph = False
checkpoint_path = None
input_graph_filename = None
# saved_model_tags = tag_constants.SERVING

freeze_graph.freeze_graph(input_graph_filename, input_saver_def_path,
                              input_binary, checkpoint_path, output_node_names,
                              restore_op_name, filename_tensor_name,
                              output_graph_filename, clear_devices, "", "", "",
                              input_meta_graph, input_saved_model_dir,)
                              # saved_model_tags)
# freeze_graph.freeze_graph(input_graph,
#              input_binary=input_binary,
#              input_saver='',
#              input_checkpoint=input_checkpoint,
#              output_node_names=output_node_names,
#              output_graph=output_graph,
#              restore_op_name='',
#              filename_tensor_name='',
#              clear_devices=True,
#              initializer_nodes='')
