import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import json
import joblib
import numpy as np

def export_isolation_forest_cpp_header(models_dir="data/models", output_header="edge/retrofit_edge_model.h"):
    """
    Exports trained Isolation Forest model, RobustScaler parameters, and thresholds into C++ header format for Uno Q.
    """
    os.makedirs(os.path.dirname(output_header), exist_ok=True)
    
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    model_path = os.path.join(models_dir, "anomaly_model.pkl")
    feats_path = os.path.join(models_dir, "selected_features.json")
    fp_path = os.path.join(models_dir, "machine_fingerprint.json")
    
    if not (os.path.exists(scaler_path) and os.path.exists(model_path)):
        print(f"[EXPORT C++] Error: Model files not found in {models_dir}")
        return None
        
    scaler = joblib.load(scaler_path)
    anomaly_model = joblib.load(model_path)
    
    with open(feats_path, 'r') as f:
        selected_features = json.load(f)
        
    with open(fp_path, 'r') as f:
        fp_data = json.load(f)
        
    # Extract sklearn Isolation Forest estimators
    if hasattr(anomaly_model, 'model'):
        sk_model = anomaly_model.model
    else:
        sk_model = anomaly_model
        
    estimators = sk_model.estimators_
    n_trees = len(estimators)
    
    # RobustScaler params
    center = scaler.center_ if hasattr(scaler, 'center_') else np.zeros(len(selected_features))
    scale = scaler.scale_ if hasattr(scaler, 'scale_') else np.ones(len(selected_features))
    
    total_nodes = 0
    max_depth_found = 0
    
    # C++ Header code generation
    cpp = []
    cpp.append("/* RetroFit Edge AI Model Header - Auto-generated for Arduino Uno Q */")
    cpp.append("#ifndef RETROFIT_EDGE_MODEL_H")
    cpp.append("#define RETROFIT_EDGE_MODEL_H")
    cpp.append("")
    cpp.append("#include <math.h>")
    cpp.append("")
    cpp.append(f"#define NUM_SELECTED_FEATURES {len(selected_features)}")
    cpp.append(f"#define NUM_TREES {n_trees}")
    cpp.append(f"#define GLOBAL_ANOMALY_THRESHOLD {fp_data.get('anomaly_threshold', 0.15):.6f}f")
    cpp.append("")
    
    # Export Scaler Parameters
    cpp.append("// RobustScaler Center (Median) and Scale (IQR)")
    cpp.append("static const float SCALER_CENTER[NUM_SELECTED_FEATURES] = { " + ", ".join([f"{v:.6f}f" for v in center]) + " };")
    cpp.append("static const float SCALER_SCALE[NUM_SELECTED_FEATURES] = { " + ", ".join([f"{v:.6f}f" for v in scale]) + " };")
    cpp.append("")
    
    # Struct for C++ Decision Node
    cpp.append("struct Node {")
    cpp.append("    int feature;")
    cpp.append("    float threshold;")
    cpp.append("    int left_child;")
    cpp.append("    int right_child;")
    cpp.append("};")
    cpp.append("")
    
    # Export Tree Arrays
    cpp.append("// Tree Decision Structures")
    for t_idx, tree in enumerate(estimators):
        t = tree.tree_
        n_nodes = t.node_count
        total_nodes += n_nodes
        
        children_left = t.children_left
        children_right = t.children_right
        feature = t.feature
        threshold = t.threshold
        
        cpp.append(f"// Tree {t_idx} ({n_nodes} nodes)")
        cpp.append(f"static const Node TREE_{t_idx}[{n_nodes}] = {{")
        for i in range(n_nodes):
            f_id = int(feature[i])
            th = float(threshold[i])
            left = int(children_left[i])
            right = int(children_right[i])
            cpp.append(f"    {{ {f_id}, {th:.6f}f, {left}, {right} }},")
        cpp.append("};")
        cpp.append("")
        
    # C++ Inference helper function
    cpp.append("static inline float compute_tree_depth(const Node* tree, int node_id, const float* x) {")
    cpp.append("    if (tree[node_id].left_child == -1) return 1.0f;")
    cpp.append("    int feat = tree[node_id].feature;")
    cpp.append("    if (x[feat] <= tree[node_id].threshold) {")
    cpp.append("        return 1.0f + compute_tree_depth(tree, tree[node_id].left_child, x);")
    cpp.append("    } else {")
    cpp.append("        return 1.0f + compute_tree_depth(tree, tree[node_id].right_child, x);")
    cpp.append("    }")
    cpp.append("}")
    cpp.append("")
    cpp.append("#endif // RETROFIT_EDGE_MODEL_H")
    
    cpp_text = "\n".join(cpp)
    with open(output_header, 'w') as f:
        f.write(cpp_text)
        
    size_kb = os.path.getsize(output_header) / 1024.0
    
    print(f"[EXPORT C++] Exported C++ Model Header to {output_header}")
    print(f" -> C++ Header File Size : {size_kb:.2f} KB")
    print(f" -> Total Decision Trees : {n_trees}")
    print(f" -> Total Tree Nodes     : {total_nodes}")
    print(f" -> Average Nodes / Tree : {total_nodes / n_trees:.1f}")
    
    return {
        'output_header': output_header,
        'size_kb': size_kb,
        'n_trees': n_trees,
        'total_nodes': total_nodes
    }

if __name__ == "__main__":
    export_isolation_forest_cpp_header()
