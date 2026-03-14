import onnx_graphsurgeon as gs
import numpy as np
import onnx

def get_nms_input(graph, class_num=2, output_name="output0"):

    tensors = graph.tensors()

    if output_name not in tensors:
        raise ValueError(f"Graph 中找不到输出张量: {output_name}")

    origin_tensor = tensors[output_name]
    print("找到输出张量:", origin_tensor)

    NUM_ANCHORS = 33600

    # Transpose [1,6,33600] -> [1,33600,6]
    output_2 = gs.Variable("output_reshape", dtype=np.float32)
    output_2_node = gs.Node(
        op="Transpose",
        inputs=[origin_tensor],
        outputs=[output_2],
        attrs={"perm": [0, 2, 1]},
    )

    # Slice -> box_input
    box_input = gs.Variable("box_input", dtype=np.float32)
    box_slice = gs.Node(
        op="Slice",
        inputs=[
            output_2,
            gs.Constant("start_box", np.array([0, 0, 0])),
            gs.Constant("end_box",   np.array([1, NUM_ANCHORS, 4])),
            gs.Constant("axes_box",  np.array([0, 1, 2])),
        ],
        outputs=[box_input],
    )

    # Slice -> raw score
    raw_score = gs.Variable("raw_score", dtype=np.float32)
    score_slice = gs.Node(
        op="Slice",
        inputs=[
            output_2,
            gs.Constant("start_s", np.array([0, 0, 4])),
            gs.Constant("end_s",   np.array([1, NUM_ANCHORS, 5])),
            gs.Constant("axes_s",  np.array([0, 1, 2])),
        ],
        outputs=[raw_score],
    )

    # Slice -> raw class id
    raw_cls = gs.Variable("raw_cls", dtype=np.float32)
    cls_slice = gs.Node(
        op="Slice",
        inputs=[
            output_2,
            gs.Constant("start_c", np.array([0, 0, 5])),
            gs.Constant("end_c",   np.array([1, NUM_ANCHORS, 6])),
            gs.Constant("axes_c",  np.array([0, 1, 2])),
        ],
        outputs=[raw_cls],
    )

    # OneHot + Mul 生成 score_input
    cls_int  = gs.Variable("cls_int",  dtype=np.int32)
    one_hot  = gs.Variable("cls_onehot", dtype=np.float32)
    score_in = gs.Variable("score_input", dtype=np.float32)

    cast_node = gs.Node("Cast", inputs=[raw_cls], outputs=[cls_int], attrs={"to": onnx.TensorProto.INT32})

    # OneHot + Mul 生成 score_input

    # Cast class_id 为 int64（OneHot 要求）
    cls_int = gs.Variable("cls_int", dtype=np.int64)
    cast_node = gs.Node(
        op="Cast",
        inputs=[raw_cls],
        outputs=[cls_int],
        attrs={"to": onnx.TensorProto.INT64},
    )

    # 正确的 ONNX OneHot（3 输入）
    # values = [off_value, on_value]
    one_hot = gs.Variable("cls_onehot", dtype=np.float32)
    onehot_node = gs.Node(
        op="OneHot",
        inputs=[
            cls_int,
            gs.Constant("depth", np.array([class_num], dtype=np.int64)),
            gs.Constant("values", np.array([0.0, 1.0], dtype=np.float32)),  # <-- 正确写法
        ],
        outputs=[one_hot],
    )

    # 每个框的 score 只属于 class_id 这一类：
    #  score_input[...,c] = score if c == class_id else 0
    score_in = gs.Variable("score_input", dtype=np.float32)
    mul_node = gs.Node(
        op="Mul",
        inputs=[raw_score, one_hot],
        outputs=[score_in],
    )


    # Register
    graph.nodes.extend([output_2_node, box_slice, score_slice, cls_slice,
                        cast_node, onehot_node, mul_node])

    # IMPORTANT: protect variables so cleanup does NOT delete them
    graph.outputs.extend([box_input, score_in])

    graph.cleanup().toposort()
    return graph

def create_and_add_plugin_node(graph, max_output_boxes=100):

    tensors = graph.tensors()
    boxes = tensors["box_input"]          # shape: [1, N, 4]
    scores = tensors["score_input"]       # shape: [1, N, num_classes]

    # -------------------------------
    # Step 1: 添加 ONNX NonMaxSuppression
    # -------------------------------
    nms_indices = gs.Variable("nms_indices", dtype=np.int64, shape=[None, 3])

    nms_node = gs.Node(
        op="NonMaxSuppression",
        inputs=[
            boxes,
            scores,
            gs.Constant("max_output_boxes", np.array([max_output_boxes], dtype=np.int64)),
            gs.Constant("iou_threshold", np.array([0.45], dtype=np.float32)),
            gs.Constant("score_threshold", np.array([0.25], dtype=np.float32)),
        ],
        outputs=[nms_indices],
        attrs={"center_point_box": 0},
    )

    graph.nodes.append(nms_node)

    # Slice 出 box_id
    box_ids = gs.Variable("selected_box_ids", dtype=np.int64, shape=[None, 1])
    slice_boxid = gs.Node(
        op="Slice",
        inputs=[
            nms_indices,
            gs.Constant("starts", np.array([0, 2])),
            gs.Constant("ends", np.array([999999, 3])),
            gs.Constant("axes", np.array([0, 1])),
            gs.Constant("steps", np.array([1, 1])),
        ],
        outputs=[box_ids],
    )
    graph.nodes.append(slice_boxid)

    # -------------------------------
    # Step 2: Gather boxes, scores
    # -------------------------------
    picked_boxes = gs.Variable("picked_boxes", dtype=np.float32, shape=[None, 4])
    gather_boxes = gs.Node(
        op="Gather",
        inputs=[boxes, box_ids],
        outputs=[picked_boxes],
        attrs={"axis": 1},
    )

    picked_scores = gs.Variable("picked_scores", dtype=np.float32, shape=[None, None])
    gather_scores = gs.Node(
        op="Gather",
        inputs=[scores, box_ids],
        outputs=[picked_scores],
        attrs={"axis": 1},
    )

    graph.nodes.extend([gather_boxes, gather_scores])

    # -------------------------------
    # Step 3: ReduceMax + ArgMax
    # -------------------------------
    final_scores = gs.Variable("final_scores", dtype=np.float32, shape=[None])
    score_max_node = gs.Node(
        op="ReduceMax",
        inputs=[picked_scores],
        outputs=[final_scores],
        attrs={"axes": [1], "keepdims": 0},
    )

    final_labels = gs.Variable("final_labels", dtype=np.int64, shape=[None])
    argmax_node = gs.Node(
        op="ArgMax",
        inputs=[picked_scores],
        outputs=[final_labels],
        attrs={"axis": 1, "keepdims": 0},
    )

    graph.nodes.extend([score_max_node, argmax_node])

    # -------------------------------
    # Step 4: 拼成 [K,5]
    # -------------------------------
    final_scores_unsq = gs.Variable("final_scores_unsq", dtype=np.float32, shape=[None, 1])
    unsq_node = gs.Node(
        op="Unsqueeze",
        inputs=[final_scores, gs.Constant("axis_s", np.array([1]))],
        outputs=[final_scores_unsq],
    )

    final_bboxes = gs.Variable(
        "final_bboxes",
        dtype=np.float32,
        shape=[None, 5]
    )
    concat_node = gs.Node(
        op="Concat",
        inputs=[picked_boxes, final_scores_unsq],
        outputs=[final_bboxes],
        attrs={"axis": 1},
    )

    graph.nodes.extend([unsq_node, concat_node])

    # -------------------------------
    # Step 5: 加 batch 维度，并固定形状
    # -------------------------------
    final_bboxes_out = gs.Variable(
        "final_bboxes_out",
        dtype=np.float32,
        shape=[1, max_output_boxes, 5]
    )
    unsq_box = gs.Node(
        op="Unsqueeze",
        inputs=[final_bboxes, gs.Constant("axis_box", np.array([0]))],
        outputs=[final_bboxes_out],
    )

    final_labels_out = gs.Variable(
        "final_labels_out",
        dtype=np.int64,
        shape=[1, max_output_boxes]
    )
    unsq_label = gs.Node(
        op="Unsqueeze",
        inputs=[final_labels, gs.Constant("axis_label", np.array([0]))],
        outputs=[final_labels_out],
    )

    graph.nodes.extend([unsq_box, unsq_label])

    graph.outputs = [final_bboxes_out, final_labels_out]

    return graph.cleanup().toposort()



def create_attrs(max_output_boxes=100):
    attrs = {}
    attrs["score_threshold"] = 0.25
    attrs["iou_threshold"] = 0.45
    attrs["max_output_boxes"] = max_output_boxes
    attrs["background_class"] = -1
    attrs["score_activation"] = False
    attrs["class_agnostic"] = True
    attrs["box_coding"] = 1
    attrs["plugin_version"] = "1"
    return attrs


if __name__ == "__main__":
    onnx_path = "pesonhead.onnx"
    graph = gs.import_onnx(onnx.load(onnx_path))

    # 根据你的模型输出配置这两个参数
    graph = get_nms_input(
        graph,
        class_num=2,
        output_name="output0",  # 换成你模型对应的输出节点名
    )

    # NMS 最大输出框数，比如 300
    graph = create_and_add_plugin_node(graph, max_output_boxes=16)

    onnx.save(gs.export_onnx(graph), "./personhead_nmsv2.onnx")
