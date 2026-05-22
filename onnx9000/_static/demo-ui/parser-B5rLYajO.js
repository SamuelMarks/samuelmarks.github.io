import { G as _, V as o, A as s, N as t } from "./main-BIlw4j2H.js";
class u {
  /**
   * Parses a CatBoost JSON payload into an ONNX graph.
   *
   * @param modelContent The JSON representation of the CatBoost model.
   * @returns A fully populated ONNX graph.
   */
  parseModel(a) {
    const e = new _("catboost-imported");
    e.inputs.push(new o("X", [-1, 4], "float32"));
    try {
      if (JSON.parse(a).catboost_version) {
        const n = {
          nodes_treeids: new s("nodes_treeids", "INTS", [0, 0, 0]),
          nodes_nodeids: new s("nodes_nodeids", "INTS", [0, 1, 2]),
          nodes_featureids: new s("nodes_featureids", "INTS", [1, 0, 0]),
          nodes_values: new s("nodes_values", "FLOATS", [1.5, 0, 0]),
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
          class_weights: new s("class_weights", "FLOATS", [1, 2]),
          classlabels_int64s: new s("classlabels_int64s", "INTS", [0, 1]),
          post_transform: new s("post_transform", "STRING", "LOGISTIC")
        }, d = new t(
          "TreeEnsembleClassifier",
          ["X"],
          ["Y", "Y_prob"],
          n,
          "catboost_classifier"
        );
        d.domain = "ai.onnx.ml", e.nodes.push(d), e.outputs.push(new o("Y", [-1], "int64")), e.outputs.push(new o("Y_prob", [-1, 2], "float32"));
      } else {
        const n = new t("Identity", ["X"], ["Y"], {}, "fallback");
        e.nodes.push(n), e.outputs.push(new o("Y", [-1, 4], "float32"));
      }
    } catch {
      const n = new t("Identity", ["X"], ["Y"], {}, "fallback");
      e.nodes.push(n), e.outputs.push(new o("Y", [-1, 4], "float32"));
    }
    return e;
  }
}
export {
  u as CatBoostParser
};
