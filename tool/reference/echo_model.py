"""A tiny ONNX graph for the Dart ONNX Runtime binding's tests: it casts its inputs to float.

Inputs `ids` (int64 [b, s]) and `mask` (bool [b, k]); outputs `ids_f = Cast(ids, float)` and
`mask_f = Cast(mask, float)`. Opset 18. Usage: python echo_model.py <out.onnx>

The file is checked with onnx.checker and run once through onnxruntime before it is written.
"""
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper


def main():
    out = sys.argv[1]
    graph = helper.make_graph(
        [
            helper.make_node("Cast", ["ids"], ["ids_f"], to=TensorProto.FLOAT),
            helper.make_node("Cast", ["mask"], ["mask_f"], to=TensorProto.FLOAT),
        ],
        "echo",
        [
            helper.make_tensor_value_info("ids", TensorProto.INT64, ["b", "s"]),
            helper.make_tensor_value_info("mask", TensorProto.BOOL, ["b", "k"]),
        ],
        [
            helper.make_tensor_value_info("ids_f", TensorProto.FLOAT, ["b", "s"]),
            helper.make_tensor_value_info("mask_f", TensorProto.FLOAT, ["b", "k"]),
        ],
    )
    # IR version 10 is what onnx 1.23 emits by default for opset 18; pin it so reruns are stable.
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10,
        producer_name="laya_server-echo",
    )
    onnx.checker.check_model(model)
    data = model.SerializeToString()

    session = ort.InferenceSession(data, providers=["CPUExecutionProvider"])
    ids_f, mask_f = session.run(None, {
        "ids": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64),
        "mask": np.array([[True, False], [False, True]]),
    })
    assert ids_f.tolist() == [[1, 2, 3], [4, 5, 6]], ids_f
    assert mask_f.tolist() == [[1, 0], [0, 1]], mask_f

    with open(out, "wb") as f:
        f.write(data)
    print(f"{len(data)} bytes, onnxruntime {ort.__version__}")


if __name__ == "__main__":
    main()
