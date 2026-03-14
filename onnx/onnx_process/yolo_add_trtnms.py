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

    onehot_node = gs.Node(
        "OneHot",
        inputs=[
            cls_int,
            gs.Constant("depth", np.array([class_num], dtype=np.int64)),
            gs.Constant("on", np.array([1.0], dtype=np.float32)),
            gs.Constant("off", np.array([0.0], dtype=np.float32)),
        ],
        outputs=[one_hot],
    )

    mul_node = gs.Node("Mul", inputs=[raw_score, one_hot], outputs=[score_in])

    # Register
    graph.nodes.extend([output_2_node, box_slice, score_slice, cls_slice,
                        cast_node, onehot_node, mul_node])

    # IMPORTANT: protect variables so cleanup does NOT delete them
    graph.outputs.extend([box_input, score_in])

    graph.cleanup().toposort()
    return graph

def create_and_add_plugin_node(graph, max_output_boxes):
    batch_size = graph.inputs[0].shape[0]
    print("The batch size is: ", batch_size)

    tensors = graph.tensors()
    boxes_tensor = tensors["box_input"]   # [B, N, 4]
    confs_tensor = tensors["score_input"] # [B, N, C]

    # EfficientNMS_TRT 原始 4 个输出
    num_detections = gs.Variable(
        name="num_detections", dtype=np.int32, shape=[batch_size, 1]
    )
    nmsed_boxes = gs.Variable(
        name="detection_boxes",
        dtype=np.float32,
        shape=[batch_size, max_output_boxes, 4],
    )
    nmsed_scores = gs.Variable(
        name="detection_scores",
        dtype=np.float32,
        shape=[batch_size, max_output_boxes],
    )
    nmsed_classes = gs.Variable(
        name="detection_classes",
        dtype=np.int32,
        shape=[batch_size, max_output_boxes],
    )

    nms_outputs = [num_detections, nmsed_boxes, nmsed_scores, nmsed_classes]

    nms_node = gs.Node(
        op="EfficientNMS_TRT",
        attrs=create_attrs(max_output_boxes),
        inputs=[boxes_tensor, confs_tensor],
        outputs=nms_outputs,
    )
    graph.nodes.append(nms_node)

    # ====== 这里开始做你想要的最终输出格式 ======
    # nmsed_boxes:   [B, max_output_boxes, 4]
    # nmsed_scores:  [B, max_output_boxes]
    # nmsed_classes: [B, max_output_boxes]

    # 1) 把 scores 变成 [B, max_output_boxes, 1]
    scores_unsq = gs.Variable(
        name="detection_scores_unsq",
        dtype=np.float32,
        shape=[batch_size, max_output_boxes, 1],
    )
    unsq_node = gs.Node(
        op="Unsqueeze",
        inputs=[nmsed_scores, gs.Constant("axis_scores", np.array([2], dtype=np.int64))],
        outputs=[scores_unsq],
    )

    # 2) concat boxes + scores -> [B, max_output_boxes, 5]
    bboxes_with_score = gs.Variable(
        name="detection_bboxes",
        dtype=np.float32,
        shape=[batch_size, max_output_boxes, 5],
    )
    concat_node = gs.Node(
        op="Concat",
        inputs=[nmsed_boxes, scores_unsq],
        outputs=[bboxes_with_score],
        attrs={"axis": 2},  # 最后一维拼 [4] + [1] = [5]
    )

    graph.nodes.extend([unsq_node, concat_node])

    # 3) 最终输出改成你要的两个：
    #    bboxes: [B, max_output_boxes, 5]
    #    labels: [B, max_output_boxes]
    final_bboxes = bboxes_with_score
    final_labels = nmsed_classes  # 直接用原来的

    graph.outputs = [final_bboxes, final_labels]

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

    onnx.save(gs.export_onnx(graph), "./yolo_nms.onnx")
