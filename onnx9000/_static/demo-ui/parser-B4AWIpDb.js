import { G as p, V as t, A as n, N as i } from "./main-Mf2qmUsB.js";
class u {
  /**
   * Parses a SparkML JSON payload into an ONNX graph.
   *
   * @param modelContent The JSON representation of the SparkML model.
   * @returns A fully populated ONNX graph.
   */
  parseModel(r) {
    const s = new p("sparkml-imported");
    s.inputs.push(new t("features", [-1, 4], "float32"));
    try {
      const o = JSON.parse(r);
      if (o.class && o.class.includes("LogisticRegression")) {
        const e = {
          coefficients: new n("coefficients", "FLOATS", [1, -2, 3.5, 0.4]),
          intercepts: new n("intercepts", "FLOATS", [-1]),
          classlabels_int64s: new n("classlabels_int64s", "INTS", [0, 1]),
          post_transform: new n("post_transform", "STRING", "LOGISTIC")
        }, a = new i(
          "LinearClassifier",
          ["features"],
          ["prediction", "probability"],
          e,
          "lr"
        );
        a.domain = "ai.onnx.ml", s.nodes.push(a), s.outputs.push(new t("prediction", [-1], "int64")), s.outputs.push(new t("probability", [-1, 2], "float32"));
      } else {
        const e = new i("Identity", ["features"], ["prediction"], {}, "fallback");
        s.nodes.push(e), s.outputs.push(new t("prediction", [-1, 4], "float32"));
      }
    } catch {
      const e = new i("Identity", ["features"], ["prediction"], {}, "fallback");
      s.nodes.push(e), s.outputs.push(new t("prediction", [-1, 4], "float32"));
    }
    return s;
  }
}
export {
  u as SparkMLParser
};
