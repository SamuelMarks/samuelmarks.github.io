import { IModelGraph, INode } from '../core/IR';

export interface IGraphLayoutNode {
  id: string;
  x: number;
  y: number;
  width: number;
  height: number;
  node: INode;
}

export interface IGraphLayoutEdge {
  source: string;
  target: string;
  points: { x: number; y: number }[];
}

export class Dagrel {
  private nodeWidth = 150;
  private nodeHeight = 50;
  private rankSeparation = 100;
  private nodeSeparation = 50;

  layout(graph: IModelGraph): { nodes: IGraphLayoutNode[]; edges: IGraphLayoutEdge[] } {
    // 1. Assign ranks (Topological sort essentially)
    const ranks = new Map<string, number>();
    const outgoingEdges = new Map<string, string[]>();
    const incomingEdges = new Map<string, string[]>();

    graph.nodes.forEach((n) => {
      outgoingEdges.set(n.name, []);
      if (!incomingEdges.has(n.name)) {
        incomingEdges.set(n.name, []);
      }
    });

    graph.nodes.forEach((n) => {
      n.inputs.forEach((input) => {
        // Find node that produces this output
        const producer = graph.nodes.find((pn) => pn.outputs.includes(input));
        if (producer) {
          if (!outgoingEdges.has(producer.name)) outgoingEdges.set(producer.name, []);
          outgoingEdges.get(producer.name)!.push(n.name);

          if (!incomingEdges.has(n.name)) incomingEdges.set(n.name, []);
          incomingEdges.get(n.name)!.push(producer.name);
        }
      });
    });

    // BFS to assign ranks
    const queue: string[] = [];
    incomingEdges.forEach((deps, nodeName) => {
      if (deps.length === 0) {
        ranks.set(nodeName, 0);
        queue.push(nodeName);
      }
    });

    while (queue.length > 0) {
      const current = queue.shift()!;
      const currentRank = ranks.get(current)!;
      const deps = outgoingEdges.get(current) || [];

      deps.forEach((dep) => {
        const nextRank = currentRank + 1;
        if (!ranks.has(dep) || ranks.get(dep)! < nextRank) {
          ranks.set(dep, nextRank);
          queue.push(dep);
        }
      });
    }

    // Unconnected nodes or cyclic graph fallback
    graph.nodes.forEach((n) => {
      if (!ranks.has(n.name)) ranks.set(n.name, 0);
    });

    // 2. Position nodes based on rank
    const layoutNodes: IGraphLayoutNode[] = [];
    const rankWidths = new Map<number, number>(); // track how many nodes in each rank

    graph.nodes.forEach((n) => {
      const rank = ranks.get(n.name)!;
      const pos = rankWidths.get(rank) || 0;

      layoutNodes.push({
        id: n.name,
        x: pos * (this.nodeWidth + this.nodeSeparation),
        y: rank * (this.nodeHeight + this.rankSeparation),
        width: this.nodeWidth,
        height: this.nodeHeight,
        node: n,
      });

      rankWidths.set(rank, pos + 1);
    });

    // Center align ranks
    const maxRankWidth = Math.max(...Array.from(rankWidths.values()));
    const maxWidth = maxRankWidth * (this.nodeWidth + this.nodeSeparation);

    layoutNodes.forEach((ln) => {
      const rank = ranks.get(ln.id)!;
      const rankCount = rankWidths.get(rank)!;
      const rankTotalWidth = rankCount * (this.nodeWidth + this.nodeSeparation);
      const xOffset = (maxWidth - rankTotalWidth) / 2;
      ln.x += xOffset;
    });

    // 3. Create edges (orthogonal/straight line stub)
    const layoutEdges: IGraphLayoutEdge[] = [];
    graph.nodes.forEach((n) => {
      const targetNode = layoutNodes.find((ln) => ln.id === n.name);
      if (!targetNode) return;

      n.inputs.forEach((input) => {
        const producerNode = graph.nodes.find((pn) => pn.outputs.includes(input));
        if (producerNode) {
          const sourceNode = layoutNodes.find((ln) => ln.id === producerNode.name);
          if (sourceNode) {
            layoutEdges.push({
              source: sourceNode.id,
              target: targetNode.id,
              points: [
                { x: sourceNode.x + sourceNode.width / 2, y: sourceNode.y + sourceNode.height },
                { x: targetNode.x + targetNode.width / 2, y: targetNode.y },
              ],
            });
          }
        }
      });
    });

    return { nodes: layoutNodes, edges: layoutEdges };
  }
}
