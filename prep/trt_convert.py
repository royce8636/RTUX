from tf2onnx import tfonnx, optimizer, tf_loader
import onnx
import onnx_graphsurgeon as gs
from onnx import shape_inference
import sys

sys.path.insert(1, "onnx_to_tf/")
from build_engine import EngineBuilder

import os
import tensorflow as tf
import argparse

def convert_to_onnx(model_dir, batch_size, input_size):
    # Load saved model
    saved_model_path = os.path.realpath(model_dir)
    assert os.path.isdir(saved_model_path)
    print(f"Converting graph from {saved_model_path} to ONNX...")
    graph_def, inputs, outputs = tf_loader.from_saved_model(saved_model_path, None, None, "serve", ["serving_default"])
    with tf.Graph().as_default() as tf_graph:
        tf.import_graph_def(graph_def, name="")
    with tf_loader.tf_session(graph=tf_graph):
        onnx_graph = tfonnx.process_tf_graph(tf_graph, input_names=inputs, output_names=outputs, opset=11)
    onnx_model = optimizer.optimize_graph(onnx_graph).make_model("Converted from {}".format(saved_model_path))


    graph = gs.import_onnx(onnx_model)
    assert graph
    print("ONNX graph created successfully")

    # Set the I/O tensor shapes
    graph.inputs[0].shape[0] = batch_size
    graph.outputs[0].shape[0] = batch_size

    if input_size and input_size > 0:
        if graph.inputs[0].shape[3] == 3:
            # Format NHWC
            graph.inputs[0].shape[1] = input_size
            graph.inputs[0].shape[2] = input_size
        elif graph.inputs[0].shape[1] == 3:
            # Format NCHW
            graph.inputs[0].shape[2] = input_size
            graph.inputs[0].shape[3] = input_size
    print("ONNX input named '{}' with shape {}".format(graph.inputs[0].name, graph.inputs[0].shape))
    print("ONNX output named '{}' with shape {}".format(graph.outputs[0].name, graph.outputs[0].shape))
    for i in range(4):
        if type(graph.inputs[0].shape[i]) != int or graph.inputs[0].shape[i] <= 0:
            print("The input shape of the graph is invalid, try overriding it by giving a fixed size with --input_size")
            sys.exit(1)

    # Fix Clip Nodes (ReLU6)
    for node in [n for n in graph.nodes if n.op == "Clip"]:
        for input in node.inputs[1:]:
            # In TensorRT, the min/max inputs on a Clip op *must* have fp32 datatype
            input.values = np.float32(input.values)

    # Run tensor shape inference
    graph.cleanup().toposort()
    model = shape_inference.infer_shapes(gs.export_onnx(graph))
    # model.inputs[0].type.tensor_type.shape.dim[0].dim_param = '?'
    graph = gs.import_onnx(model)

    # Save updated model
    graph.cleanup().toposort()
    model = gs.export_onnx(graph)
    onnx_path = os.path.realpath(f"{model_dir}.onnx")
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    onnx.save(model, onnx_path)
    print("ONNX model saved to {}".format(onnx_path))
    return onnx_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert TensorFlow model to ONNX")
    parser.add_argument("--model_dir", "-d", type=str, required=True, help="Path to the saved model directory")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for the model")
    parser.add_argument("--input_size", type=int, default=None, help="Input size for the model")
    args = parser.parse_args()

    onnx_path = convert_to_onnx(args.model_dir, args.batch_size, args.input_size)

    directory = os.path.dirname(onnx_path)
    model_name = os.path.basename(onnx_path).replace(".onnx", ".trt")
    engine_dir = os.path.join(directory, model_name)

    print(f"Saving TensorRT engine to {engine_dir}")

    builder = EngineBuilder(False)
    builder.create_network(onnx_path)
    builder.create_engine(
        engine_dir,
        # f"{args['directory']}/{args['game']}/models/{model_name}.trt",
        'fp32',
        calib_input=None,
        calib_cache="./calibration_cache",
        calib_batch_size=1,
        calib_preprocessor="V2",
        calib_timing_cache="./timing_cache",
    )