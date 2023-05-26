import onnx

model = onnx.load("models/20230511131658_steering-angle/last.onnx")
model.graph.input[0].type.tensor_type.shape.dim[0].dim_value = 1
model.graph.output[0].type.tensor_type.shape.dim[0].dim_value = 1
# model.save("models/20230511131658_steering-angle/last_converted.onnx")
onnx.save(model, "models/20230511131658_steering-angle/last_converted.onnx")
