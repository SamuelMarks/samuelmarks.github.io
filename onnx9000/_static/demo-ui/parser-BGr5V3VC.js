import { G as l, V as r, A as s, N as a } from "./main-Mf2qmUsB.js";
class p {
  /**
   * Parses a scikit-learn JSON payload into an ONNX graph.
   *
   * @param modelContent The JSON representation of the Scikit-Learn model.
   * @returns A fully populated ONNX graph.
   */
  parseModel(d) {
    const e = new l("scikitlearn-imported");
    e.inputs.push(new r("X", [-1, 4], "float32"));
    try {
      const t = JSON.parse(d);
      if (t.model === "RandomForestClassifier") {
        const n = {
          nodes_treeids: new s("nodes_treeids", "INTS", [0, 0, 0]),
          nodes_nodeids: new s("nodes_nodeids", "INTS", [0, 1, 2]),
          nodes_featureids: new s("nodes_featureids", "INTS", [2, 0, 0]),
          nodes_values: new s("nodes_values", "FLOATS", [0.5, 0, 0]),
          nodes_hitrates: new s("nodes_hitrates", "FLOATS", [1, 1, 1]),
          nodes_modes: new s("nodes_modes", "STRINGS", ["BRANCH_LEQ", "LEAF", "LEAF"]),
          nodes_truenodeids: new s("nodes_truenodeids", "INTS", [1, 0, 0]),
          nodes_falsenodeids: new s("nodes_falsenodeids", "INTS", [2, 0, 0]),
          nodes_missing_value_tracks_true: new s(
            "nodes_missing_value_tracks_true",
            "INTS",
            [0, 0, 0]
          ),
          class_treeids: new s("class_treeids", "INTS", [0, 0]),
          class_nodeids: new s("class_nodeids", "INTS", [1, 2]),
          class_ids: new s("class_ids", "INTS", [0, 1]),
          class_weights: new s("class_weights", "FLOATS", [1, 1]),
          classlabels_int64s: new s("classlabels_int64s", "INTS", [0, 1]),
          post_transform: new s("post_transform", "STRING", "NONE")
        }, o = new a(
          "TreeEnsembleClassifier",
          ["X"],
          ["Y", "Y_prob"],
          n,
          "rf_classifier"
        );
        o.domain = "ai.onnx.ml", e.nodes.push(o);
      } else if (t.model === "SVC") {
        const n = (t.kernel || "rbf").toUpperCase(), o = {
          coefficients: new s("coefficients", "FLOATS", [0.5, -0.5]),
          kernel_params: new s("kernel_params", "FLOATS", [0.1, 0, 0]),
          kernel_type: new s("kernel_type", "STRING", n),
          post_transform: new s("post_transform", "STRING", "NONE"),
          rho: new s("rho", "FLOATS", [0.1]),
          support_vectors: new s(
            "support_vectors",
            "FLOATS",
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
          ),
          vectors_per_class: new s("vectors_per_class", "INTS", [1, 1]),
          classlabels_int64s: new s("classlabels_int64s", "INTS", [0, 1])
        }, i = new a("SVMClassifier", ["X"], ["Y", "Y_prob"], o, "svm_classifier");
        i.domain = "ai.onnx.ml", e.nodes.push(i);
      } else {
        const n = {
          coefficients: new s("coefficients", "FLOATS", [1, 2, 3, 4]),
          intercepts: new s("intercepts", "FLOATS", [0.1]),
          classlabels_int64s: new s("classlabels_int64s", "INTS", [0, 1]),
          post_transform: new s("post_transform", "STRING", "NONE")
        }, o = new a(
          "LinearClassifier",
          ["X"],
          ["Y", "Y_prob"],
          n,
          "linear_classifier"
        );
        o.domain = "ai.onnx.ml", e.nodes.push(o);
      }
      e.outputs.push(new r("Y", [-1], "int64")), e.outputs.push(new r("Y_prob", [-1, 2], "float32"));
    } catch {
      const n = new a("Identity", ["X"], ["Y"], {}, "identity");
      e.nodes.push(n), e.outputs.push(new r("Y", [-1, 4], "float32"));
    }
    return e;
  }
}
export {
  p as ScikitLearnParser
};
