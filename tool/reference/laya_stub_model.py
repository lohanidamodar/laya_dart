"""A tiny ONNX graph with the Laya decision graph's signature, for OrtBackend's tests.

It has the onnx-community export's input names and types -- `input_ids` int64 [b, s],
`attention_mask` int64 [b, s], `marker_pos` int64 [b, k], `marker_mask` bool [b, k] and `qtype`
int64 [b] -- and its outputs `logits` float [b, k] and `act_logits` float [b, 2]. The values are
arithmetic over every input, so a test can tell that each input arrived where it belongs:

    logits[i, j]     = marker_pos[i, j] + 100 * qtype[i]
    act_logits[i, 0] = sum_s input_ids[i, s] * attention_mask[i, s]
    act_logits[i, 1] = sum_j marker_mask[i, j]

It is not a model. Opset 18. Usage: python laya_stub_model.py <out.onnx>

The file is checked with onnx.checker and run once through onnxruntime before it is written.
"""
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper


def main():
    out = sys.argv[1]
    i64 = TensorProto.INT64
    f32 = TensorProto.FLOAT
    nodes = [
        helper.make_node("Constant", [], ["one"], value=helper.make_tensor("one", i64, [1], [1])),
        helper.make_node("Constant", [], ["hundred"], value=helper.make_tensor("h", i64, [], [100])),
        # logits = marker_pos + 100 * qtype[:, None]
        helper.make_node("Unsqueeze", ["qtype", "one"], ["qtype_col"]),
        helper.make_node("Mul", ["qtype_col", "hundred"], ["qtype_100"]),
        helper.make_node("Add", ["marker_pos", "qtype_100"], ["logits_i"]),
        helper.make_node("Cast", ["logits_i"], ["logits"], to=f32),
        # act_logits = [sum(input_ids * attention_mask), sum(marker_mask)]
        helper.make_node("Mul", ["input_ids", "attention_mask"], ["ids_kept"]),
        helper.make_node("ReduceSum", ["ids_kept", "one"], ["ids_sum"], keepdims=1),
        helper.make_node("Cast", ["marker_mask"], ["mask_i"], to=i64),
        helper.make_node("ReduceSum", ["mask_i", "one"], ["mask_sum"], keepdims=1),
        helper.make_node("Concat", ["ids_sum", "mask_sum"], ["act_i"], axis=1),
        helper.make_node("Cast", ["act_i"], ["act_logits"], to=f32),
    ]
    graph = helper.make_graph(
        nodes,
        "laya_stub",
        [
            helper.make_tensor_value_info("input_ids", i64, ["b", "s"]),
            helper.make_tensor_value_info("attention_mask", i64, ["b", "s"]),
            helper.make_tensor_value_info("marker_pos", i64, ["b", "k"]),
            helper.make_tensor_value_info("marker_mask", TensorProto.BOOL, ["b", "k"]),
            helper.make_tensor_value_info("qtype", i64, ["b"]),
        ],
        [
            helper.make_tensor_value_info("logits", f32, ["b", "k"]),
            helper.make_tensor_value_info("act_logits", f32, ["b", 2]),
        ],
    )
    # IR version 10 is what onnx 1.23 emits by default for opset 18; pin it so reruns are stable.
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10,
        producer_name="laya_server-laya-stub",
    )
    onnx.checker.check_model(model, full_check=True)
    data = model.SerializeToString()

    session = ort.InferenceSession(data, providers=["CPUExecutionProvider"])
    logits, act = session.run(None, {
        "input_ids": np.array([[5, 6, 7], [8, 9, 1]], dtype=np.int64),
        "attention_mask": np.array([[1, 1, 1], [1, 1, 0]], dtype=np.int64),
        "marker_pos": np.array([[1, 2], [1, 0]], dtype=np.int64),
        "marker_mask": np.array([[True, True], [True, False]]),
        "qtype": np.array([0, 2], dtype=np.int64),
    })
    assert logits.tolist() == [[1, 2], [201, 200]], logits
    assert act.tolist() == [[18, 2], [17, 1]], act

    with open(out, "wb") as f:
        f.write(data)
    print(f"{len(data)} bytes, onnxruntime {ort.__version__}")


if __name__ == "__main__":
    main()
