import tensorflow as tf
# import cv2
import numpy as np
from PIL import Image
from preprocess_data import *
# from ssdoil import *
# from tensorflow.python.tools.import_pb_to_tensorboard import import_to_tensorboard

# import_to_tensorboard("./tf_models/saved_model.pb", "tb_log")

imagenet_stats = (np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225]))


def normalize(image):
	# image = (image - imagenet_stats[0][])  / imagenet_stats[1]
	mean = imagenet_stats[0]
	std = imagenet_stats[1]

	for i in range(image.shape[0]):
		image[i] = (image[i] - mean[i]) / std[i]
	return image#(image - mean[:, None, None]) / std[:, None, None] 
	# (image - mean[:, None, None]) / std[:, None, None]


def resize(image):
	# image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
	image = cv2.resize(image, (224, 224), cv2.INTER_LINEAR).transpose(2, 0, 1)
	return image

def preprocess_image(path):
	image = cv2.imread(path)
	image = resize(image)
	image = normalize(image)
	# Making channel first
	return image

path = './Bosch_trafficlight_data/TESTJPEGS/test1.jpg'

# im = preprocess_image('./Bosch_trafficlight_data/TESTJPEGS/test1.jpg')

# print('CV image')
# cv_image = cv2.imread(path)
# cv_image = cv2.cvtColor(cv_image,cv2.COLOR_BGR2RGB)
# # cv_image = resize(cv_image)
# cv_image = cv_image
# print(cv_image.dtype)
# print(cv_image[0][0][0])
# cv_image = cv_image / 255.0
# print(cv_image[0][0])
# print(cv_image.dtype)
# cv_image = resize(cv_image)
# print(cv_image[0][0])
# print(cv_image.dtype)



print('PIL image')
pil_image = Image.open(path)
# print('=====================')
# print(np.asarray(pil_image)[0][0])
pil_image = pil_image.resize((224, 224), resample=Image.BILINEAR)

pil_image = np.asarray(pil_image).transpose(2, 0, 1)
# print('After Resizing')
# print('====================')
# print(pil_image[0])
# print(pil_image.shape)
pil_image = np.multiply(pil_image, 1.0 /255.0)
# print('After Standarization')
# print('=====================')
# print(pil_image[0][0])
pil_image = normalize(pil_image)
# print('After Normalization')
# print('=====================')
# print(pil_image[0][0])

# test_tensor = tf.constant(np.expand_dims(pil_image, axis=0), dtype=tf.float32)
# c = tf.image.resize_bilinear(b, (224, 224))

# with tf.Session() as sess:
	
# 	print(test_tensor)
# 	sess.close()



# print(im)
# print(im.shape)


from tensorflow.python.platform import gfile
GRAPH_PB_PATH = './tf_models/saved_model.pb'

# Load Tfmodel 
with tf.Session() as sess:
	print('load graph')
	with gfile.FastGFile(GRAPH_PB_PATH, 'rb') as f:
		graph_def = tf.GraphDef()
	graph_def.ParseFromString(f.read())
	sess.graph.as_default()
	tf.import_graph_def(graph_def, name='')
	graph_nodes = [n for n in graph_def.node]
	names = []
	for t in graph_nodes:
		names.append(t.name)
	print('--------------------ops----------------')
	op = sess.graph.get_operations()
	# for m in op:
	print(op[0].values())
	print(op[-3].values())
	print(op[-1].values())

	print('--------------------end ops----------------')

	input_x = sess.graph.get_tensor_by_name("0:0")
	outputs1 = sess.graph.get_tensor_by_name("concat_52:0")
	outputs2 = sess.graph.get_tensor_by_name("concat_53:0")

	output_tf_pb = sess.run([outputs1, outputs2], feed_dict={input_x: np.expand_dims(pil_image, axis=0)})
	print('output_tf_pb = {} '.format(output_tf_pb))
	print('outputs shapes = {} and {} '.format(len(output_tf_pb[0][0]), len(output_tf_pb[1][0])))


	print('---------------Create BBOX----------------')
	bbox_analyzer = BboxAnalyzer()
	print(sess.run(bbox_analyzer._grid_sizes))
	print(sess.run(bbox_analyzer._anchors))
	print(sess.run(bbox_analyzer._anchor_cnr))

	# input_x = sess.graph.get_tensor_by_name("Const_424:0") # input
    # outputs1 = sess.graph.get_tensor_by_name('concat_52:0') # 5
    # outputs2 = sess.graph.get_tensor_by_name('concat_53:0') # 10
    # output_tf_pb = sess.run([outputs1, outputs2], feed_dict={input_x:pil_image})
    #output_tf_pb = sess.run([outputs1, outputs2], feed_dict={input_x:np.random.randn(1, 3, 224, 224)})
    # print('output_tf_pb = {}'.format(output_tf_pb))

	# input: Const:0
	# Const_424:0
	# output: concat_52:0, concat_53:0
	# print(names)
	# print(graph_nodes[0].name)
	# print(graph_nodes[-3].name)
	# print(graph_nodes[-1].name)


# 	tf.saved_model.loader.load(sess, './tf_models')
# 	# Restore 
# 	graph = tf.get_default_graph()
# 	print(graph.get_operations())
	# print('Model reloaded')

