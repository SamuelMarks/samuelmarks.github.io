class t {
  /**
   * Parses a PaddlePaddle model description.
   * @param modelJson The model JSON string or object.
   * @returns The parsed model object.
   */
  parseModel(r) {
    if (typeof r == "string")
      try {
        return JSON.parse(r);
      } catch {
        return { blocks: [] };
      }
    return r;
  }
  /**
   * Parses PaddlePaddle binary weights.
   * @param weightsBuffer The binary weights buffer.
   * @returns An object representing the parsed weights.
   */
  parseWeights(r) {
    return {
      byteLength: r.byteLength
    };
  }
}
export {
  t as PaddleParser
};
