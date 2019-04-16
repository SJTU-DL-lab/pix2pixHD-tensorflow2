import tensorflow as tf
from tensorflow.python.eager.function import defun

W = tf.Variable(
  tf.keras.initializers.glorot_uniform()(
    (10, 10)))

@tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
def f(x):
  return tf.matmul(x, W)

# Retrieve the object corresponding to
# a particular input signature:
graph = f.get_concrete_function().graph
graph_def = graph.as_graph_def()
print([node.name for node in graph_def.node])
print(graph_def)
#
# tf.io.write_graph(graph_def, './', 'graph1.pb', False)
# with open("./graph.pb", "wb") as f:
#   f.write(graph_def.SerializeToString())
