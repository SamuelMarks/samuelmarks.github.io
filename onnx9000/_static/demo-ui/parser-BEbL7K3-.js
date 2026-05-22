import { G as a, V as t, N as d, A as e } from "./main-Mf2qmUsB.js";
class u {
  /**
   * Parses a LightGBM model string into an ONNX graph.
   *
   * @param modelContent The raw string representation of the LightGBM model.
   * @returns A fully populated ONNX graph.
   */
  parseModel(r) {
    const s = new a("lightgbm-imported");
    if (s.inputs.push(new t("X", [-1, 4], "float32")), r.includes("tree")) {
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
      }, o = new d("TreeEnsembleRegressor", ["X"], ["Y"], n, "lgbm_regressor");
      o.domain = "ai.onnx.ml", s.nodes.push(o), s.outputs.push(new t("Y", [-1, 1], "float32"));
    } else {
      const n = new d("Identity", ["X"], ["Y"], {}, "fallback");
      s.nodes.push(n), s.outputs.push(new t("Y", [-1, 4], "float32"));
    }
    return s;
  }
}
export {
  u as LightGBMParser
};
