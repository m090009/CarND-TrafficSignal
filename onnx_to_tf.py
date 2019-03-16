import onnx
from onnx_tf.backend import prepare

# Load onnx model
onnx_model = onnx.load("model_715.onnx")
# Prepare tf represenation
tf_rep = prepare(onnx_model, strict=False)
# Export the model to tesnsorflow
tf_rep.export_graph("tensorflow_model_715")