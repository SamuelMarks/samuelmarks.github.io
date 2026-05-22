import { IModelGraph, INode } from '../core/IR';

export type ExecutionProvider = 'CPU' | 'WebGPU' | 'WebNN' | 'WASM';

export interface IPartitionedGraph extends IModelGraph {
  partitions: Map<string, ExecutionProvider | string>; // Node Name -> Provider OR Peer ID
}

/**
 * 511. Map specific graph partitions to specific execution providers explicitly.
 * 512. Automatically fallback to CPU (WASM) for unsupported ops.
 */
export class GraphPartitioner {
  public static partition(graph: IModelGraph, prefer: ExecutionProvider): IPartitionedGraph {
    const partitionedGraph: IPartitionedGraph = {
      ...JSON.parse(JSON.stringify(graph)),
      partitions: new Map(),
    };

    // Re-attach rawData
    for (let i = 0; i < graph.initializers.length; i++) {
      if (graph.initializers[i].rawData) {
        partitionedGraph.initializers[i].rawData = graph.initializers[i].rawData;
      }
    }

    partitionedGraph.nodes.forEach((node) => {
      const ep = this.selectBestProvider(node, prefer);
      partitionedGraph.partitions.set(node.name, ep);
    });

    return partitionedGraph;
  }

  /**
   * 372. Implement graph partitioning: split IModelGraph into subgraphs across peers
   * 374. Assign heavier subgraphs (MatMul/Conv) to WebGPU peers, lighter to WASM peers
   * 378. Pipeline Parallelism representation
   */
  public static partitionSwarm(
    graph: IModelGraph,
    peers: { id: string; compute: 'High' | 'Low' }[],
  ): IPartitionedGraph {
    const partitionedGraph: IPartitionedGraph = {
      ...JSON.parse(JSON.stringify(graph)),
      partitions: new Map(),
    };

    // Re-attach rawData
    for (let i = 0; i < graph.initializers.length; i++) {
      if (graph.initializers[i].rawData)
        partitionedGraph.initializers[i].rawData = graph.initializers[i].rawData;
    }

    if (peers.length === 0) return partitionedGraph; // Fallback local

    const highComputePeers = peers.filter((p) => p.compute === 'High').map((p) => p.id);
    const lowComputePeers = peers.filter((p) => p.compute === 'Low').map((p) => p.id);

    let pIdx = 0;

    // Simple topological assignment for Pipeline Parallelism (378)
    // 384. Tensor Parallelism stub (Splitting MatMul across peers is simulated here by tagging nodes)
    partitionedGraph.nodes.forEach((node) => {
      const type = node.opType;
      let assignedPeer = peers[0].id;

      // 384. Tensor parallelism mock
      if (type === 'MatMul' && peers.length > 1) {
        // Instead of a single peer, we assign a "Split_Cluster" tag which WebNN provider
        // interprets as "chunk matrix A, send to Peer 1 & 2, wait for chunks, concat"
        assignedPeer = `Cluster_${peers[0].id}_${peers[1].id}`;
      }

      if (['MatMul', 'Conv', 'Gemm'].includes(type) && highComputePeers.length > 0) {
        // Route heavy ops to high compute peers
        assignedPeer = highComputePeers[pIdx % highComputePeers.length];
        pIdx++;
      } else if (lowComputePeers.length > 0) {
        assignedPeer = lowComputePeers[pIdx % lowComputePeers.length];
        pIdx++;
      } else {
        assignedPeer = peers[pIdx % peers.length].id;
        pIdx++;
      }

      partitionedGraph.partitions.set(node.name, assignedPeer);
    });

    return partitionedGraph;
  }

  /**
   * 380. Handle peer disconnects gracefully by re-assigning their subgraph
   */
  public static handlePeerDisconnect(
    partitionedGraph: IPartitionedGraph,
    lostPeerId: string,
    remainingPeers: string[],
  ): IPartitionedGraph {
    const repairedGraph = {
      ...JSON.parse(JSON.stringify(partitionedGraph)),
      partitions: new Map(partitionedGraph.partitions),
    } as IPartitionedGraph;

    // Re-attach rawData
    for (let i = 0; i < partitionedGraph.initializers.length; i++) {
      if (partitionedGraph.initializers[i].rawData)
        repairedGraph.initializers[i].rawData = partitionedGraph.initializers[i].rawData;
    }

    const newTarget = remainingPeers.length > 0 ? remainingPeers[0] : 'LocalFallback';

    // 390. Implement a fault-tolerant fallback (If 0 peers, we reroute to native LocalFallback)
    repairedGraph.partitions.forEach((assignee, nodeName) => {
      if (assignee === lostPeerId) {
        console.warn(
          `[Swarm] Re-assigning orphaned node ${nodeName} from ${lostPeerId} to ${newTarget}`,
        );
        repairedGraph.partitions.set(nodeName, newTarget);
      }
    });

    return repairedGraph;
  }

  private static selectBestProvider(node: INode, prefer: ExecutionProvider): ExecutionProvider {
    // 512. Automatic Fallback Logic

    // WebNN unsupported mock list (e.g. NonZero, Loop, Scan, certain Reshapes with dynamic shapes)
    const webnnUnsupported = ['NonZero', 'Loop', 'Scan', 'If', 'CustomOp'];

    // WebGPU unsupported mock list (e.g. string operations, complex control flow)
    const webgpuUnsupported = ['StringNormalizer', 'RegexSplit', 'Loop', 'If'];

    if (prefer === 'WebNN' && webnnUnsupported.includes(node.opType)) {
      console.warn(`[Partitioner] ${node.opType} unsupported on WebNN. Falling back to CPU/WASM.`);
      return 'WASM';
    }

    if (prefer === 'WebGPU' && webgpuUnsupported.includes(node.opType)) {
      console.warn(`[Partitioner] ${node.opType} unsupported on WebGPU. Falling back to CPU/WASM.`);
      return 'WASM';
    }

    // 513. Ping-ponging between providers (Host-to-Device / Device-to-Host)
    // To optimize 515, if a subgraph of 3 nodes is [WebGPU, WASM, WebGPU],
    // it might be cheaper to run the WASM node on WebGPU (if possible) or vice-versa
    // to avoid D2H and H2D copy overheads.
    // This stub represents the initial naive mapping.

    return prefer;
  }
}
