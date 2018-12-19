import tensorflow as tf
from tensorflow.python.tools import freeze_graph

model_name = 'yolov3'
input_graph_path = 'model_graph/yolov3.pbtxt'
checkpoint_path = 'checkpoints/yolov3'
input_binary = False

input_saver_def_path = ""

#output_node_names = "concat_11/concat_dim, concat_12/concat_dim, concat_13/concat_dim"
output_node_names = "boxes, scores, classes" 
restore_op_name = "save/restore_all"
filename_tensor_name = ""
output_frozen_graph_name = 'frozen_' + model_name + '.pb'
output_optimized_graph_name = 'optimized_' + model_name + '.pb'
clear_devices = True

freeze_graph.freeze_graph(input_graph_path, input_saver_def_path, input_binary, checkpoint_path, output_node_names, 
        restore_op_name, filename_tensor_name, output_frozen_graph_name, clear_devices, "")

