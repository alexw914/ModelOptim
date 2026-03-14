import onnx
import onnx_graphsurgeon as gs
import numpy as np

def convert_gemm_to_matmul_add_preserve_graph(graph):
    for node in list(graph.nodes):
        if node.op == "Gemm":
            print(f"Processing Gemm node: {node.name}")
            
            A = node.inputs[0]
            B = node.inputs[1]
            bias = node.inputs[2] if len(node.inputs) > 2 else None
            Y = node.outputs[0]

            alpha = node.attrs.get("alpha", 1.0)
            beta = node.attrs.get("beta", 1.0)
            transB = node.attrs.get("transB", 0)

            matmul_A = A
            matmul_B = B

            # Step 2: Transpose B if needed
            if transB == 1:
                transposed_B = gs.Variable(name=f"{B.name}_transposed", dtype=None)
                trans_node = gs.Node(op="Transpose", inputs=[B], outputs=[transposed_B], attrs={"perm": [1, 0]})
                graph.nodes.append(trans_node)
                matmul_B = transposed_B

            # Step 3: Create MatMul node
            matmul_out = gs.Variable(name=f"{node.name}_matmul_out", dtype=None)
            matmul_node = gs.Node(op="MatMul", inputs=[matmul_A, matmul_B], outputs=[matmul_out])
            graph.nodes.append(matmul_node)

            # Step 4: Create Add node if bias exists
            final_out = matmul_out
            if bias is not None:
                add_out = gs.Variable(name=f"{node.name}_add_out", dtype=None)
                add_node = gs.Node(op="Add", inputs=[matmul_out, bias], outputs=[add_out])
                graph.nodes.append(add_node)
                final_out = add_out

            # Step 5: Redirect consumers of original Gemm output Y
            for consumer in Y.outputs:
                for idx, input_ in enumerate(consumer.inputs):
                    if input_ is Y:
                        consumer.inputs[idx] = final_out

            # Step 6: If Y is a graph output, patch it
            for idx, graph_output in enumerate(graph.outputs):
                if graph_output is Y:
                    graph.outputs[idx] = final_out

            # Step 7: Remove the original Gemm node
            graph.nodes.remove(node)

    return graph

# 主流程
model_path = "helmetcolor_cls_v0_0_0_0_int8.onnx"
graph = gs.import_onnx(onnx.load(model_path))
graph = convert_gemm_to_matmul_add_preserve_graph(graph)
graph.cleanup().toposort()
onnx_model = gs.export_onnx(graph)
onnx_model.ir_version = 9 
onnx.save(onnx_model, "matmul_" + model_path)
