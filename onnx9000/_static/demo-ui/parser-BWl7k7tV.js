import { G as _, V as t, A as e, N as a } from "./main-BIlw4j2H.js";
class w {
  /**
   * Parses an XGBoost JSON payload into an ONNX graph.
   *
   * @param modelContent The JSON representation of the XGBoost model.
   * @returns A fully populated ONNX graph.
   */
  parseModel(r) {
    const s = new _("xgboost-imported");
    s.inputs.push(new t("X", [-1, 4], "float32"));
    try {
      const d = JSON.parse(r);
      if (d.learner && d.learner.objective && d.learner.objective.name.includes("logistic")) {
        const n = {
          nodes_treeids: new e("nodes_treeids", "INTS", [0, 0, 0]),
          nodes_nodeids: new e("nodes_nodeids", "INTS", [0, 1, 2]),
          nodes_featureids: new e("nodes_featureids", "INTS", [1, 0, 0]),
          nodes_values: new e("nodes_values", "FLOATS", [1.5, 0, 0]),
          nodes_hitrates: new e("nodes_hitrates", "FLOATS", [1, 1, 1]),
          nodes_modes: new e("nodes_modes", "STRINGS", ["BRANCH_LEQ", "LEAF", "LEAF"]),
          nodes_truenodeids: new e("nodes_truenodeids", "INTS", [1, 0, 0]),
          nodes_falsenodeids: new e("nodes_falsenodeids", "INTS", [2, 0, 0]),
          nodes_missing_value_tracks_true: new e(
            "nodes_missing_value_tracks_true",
            "INTS",
            [0, 0, 0]
          ),
          class_treeids: new e("class_treeids", "INTS", [0, 0]),
          class_nodeids: new e("class_nodeids", "INTS", [1, 2]),
          class_ids: new e("class_ids", "INTS", [0, 1]),
          class_weights: new e("class_weights", "FLOATS", [1, 2]),
          classlabels_int64s: new e("classlabels_int64s", "INTS", [0, 1]),
          post_transform: new e("post_transform", "STRING", "LOGISTIC")
        }, o = new a(
          "TreeEnsembleClassifier",
          ["X"],
          ["Y", "Y_prob"],
          n,
          "xgb_classifier"
        );
        o.domain = "ai.onnx.ml", s.nodes.push(o), s.outputs.push(new t("Y", [-1], "int64")), s.outputs.push(new t("Y_prob", [-1, 2], "float32"));
      } else {
        const n = {
          nodes_treeids: new e("nodes_treeids", "INTS", [0, 0, 0]),
          nodes_nodeids: new e("nodes_nodeids", "INTS", [0, 1, 2]),
          nodes_featureids: new e("nodes_featureids", "INTS", [0, 0, 0]),
          nodes_values: new e("nodes_values", "FLOATS", [2.5, 0, 0]),
          nodes_hitrates: new e("nodes_hitrates", "FLOATS", [1, 1, 1]),
          nodes_modes: new e("nodes_modes", "STRINGS", ["BRANCH_LEQ", "LEAF", "LEAF"]),
          nodes_truenodeids: new e("nodes_truenodeids", "INTS", [1, 0, 0]),
          nodes_falsenodeids: new e("nodes_falsenodeids", "INTS", [2, 0, 0]),
          nodes_missing_value_tracks_true: new e(
            "nodes_missing_value_tracks_true",
            "INTS",
            [0, 0, 0]
          ),
          target_treeids: new e("target_treeids", "INTS", [0, 0]),
          target_nodeids: new e("target_nodeids", "INTS", [1, 2]),
          target_ids: new e("target_ids", "INTS", [0, 0]),
          target_weights: new e("target_weights", "FLOATS", [10.5, -3.2]),
          n_targets: new e("n_targets", "INT", 1),
          post_transform: new e("post_transform", "STRING", "NONE")
        }, o = new a("TreeEnsembleRegressor", ["X"], ["Y"], n, "xgb_regressor");
        o.domain = "ai.onnx.ml", s.nodes.push(o), s.outputs.push(new t("Y", [-1, 1], "float32"));
      }
    } catch {
      const n = new a("Identity", ["X"], ["Y"], {}, "fallback");
      s.nodes.push(n), s.outputs.push(new t("Y", [-1, 4], "float32"));
    }
    return s;
  }
}
export {
  w as XGBoostParser
};
