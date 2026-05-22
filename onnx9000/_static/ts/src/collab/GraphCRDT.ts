import { IModelGraph, INode } from '../core/IR';
import { globalEvents } from '../core/State';

// 521. Generic CRDT model for IModelGraph
// A true CRDT for JSON involves Lamport timestamps and logical clocks.
// This is a minimal LWW (Last-Write-Wins) Map stub applied to graph nodes.
export class GraphCRDT {
  private localClock = 0;
  private peerClocks = new Map<string, number>();
  private nodeMap = new Map<
    string,
    { node: INode; ts: number; peerId: string; deleted: boolean }
  >();

  private modelRef: IModelGraph | null = null;
  private localPeerId: string;

  // 535. Undo/Redo tracking
  private historyStack: any[] = [];
  private redoStack: any[] = [];

  // 532. Granular permissions
  public role: 'Admin' | 'Edit' | 'View' = 'Admin';
  // 533. Lock specific subgraphs
  public lockedNodes: Set<string> = new Set();

  // 534. Offline edits sync queue
  private pendingDeltas: any[] = [];

  constructor(peerId: string) {
    this.localPeerId = peerId;
  }

  init(model: IModelGraph): void {
    this.modelRef = model;
    this.localClock++;

    // Load initial state into CRDT
    model.nodes.forEach((n) => {
      this.nodeMap.set(n.name, {
        node: JSON.parse(JSON.stringify(n)),
        ts: this.localClock,
        peerId: this.localPeerId,
        deleted: false,
      });
    });
  }

  // 524. Local mutations
  deleteNode(nodeName: string): any {
    if (this.role === 'View') throw new Error('Permission Denied: View Only');
    if (this.lockedNodes.has(nodeName) && this.role !== 'Admin')
      throw new Error('Permission Denied: Node is locked by Admin');

    if (!this.nodeMap.has(nodeName)) return null;
    this.localClock++;
    const state = this.nodeMap.get(nodeName)!;
    state.deleted = true;
    state.ts = this.localClock;
    state.peerId = this.localPeerId;

    this.syncToModel();
    return this.createDelta(nodeName, state);
  }

  updateNode(node: INode): any {
    if (this.role === 'View') throw new Error('Permission Denied: View Only');
    if (this.lockedNodes.has(node.name) && this.role !== 'Admin')
      throw new Error('Permission Denied: Node is locked by Admin');

    this.localClock++;
    this.nodeMap.set(node.name, {
      node: JSON.parse(JSON.stringify(node)),
      ts: this.localClock,
      peerId: this.localPeerId,
      deleted: false,
    });

    this.syncToModel();
    return this.createDelta(node.name, this.nodeMap.get(node.name)!);
  }

  // Handle incoming remote syncs
  // 525. Handle concurrent edits (LWW logic)
  applyDelta(delta: any): boolean {
    const { nodeName, node, ts, peerId, deleted } = delta;

    // Update peer clock watermark
    const currentPeerClock = this.peerClocks.get(peerId) || 0;
    if (ts > currentPeerClock) {
      this.peerClocks.set(peerId, ts);
    }

    const existing = this.nodeMap.get(nodeName);

    // Last-Write-Wins (LWW) conflict resolution
    // If our local timestamp is older, OR if timestamps match but remote peerId > localId (arbitrary tie-breaker)
    if (!existing || ts > existing.ts || (ts === existing.ts && peerId > existing.peerId)) {
      this.nodeMap.set(nodeName, { node, ts, peerId, deleted });
      this.syncToModel();
      return true; // Graph changed
    }

    return false; // Ignored (stale)
  }

  private createDelta(nodeName: string, state: any): any {
    const delta = {
      type: 'crdt_update',
      nodeName,
      node: state.node,
      ts: state.ts,
      peerId: state.peerId,
      deleted: state.deleted,
    };

    this.historyStack.push(delta);
    this.pendingDeltas.push(delta);

    return delta;
  }

  // 534. Get pending deltas for reconnection
  public flushPending(): any[] {
    const p = [...this.pendingDeltas];
    this.pendingDeltas = [];
    return p;
  }

  // 535. Undo Stub
  public undo(): void {
    if (this.historyStack.length === 0) return;
    const lastDelta = this.historyStack.pop();
    // We need to calculate the inverse operation here and apply it
    // For mock purposes:
    console.log('Undoing CRDT delta', lastDelta);
  }

  // 533. Admin Locking
  public lockSubgraph(nodes: string[]): void {
    if (this.role === 'Admin') {
      nodes.forEach((n) => this.lockedNodes.add(n));
    }
  }

  // 546. Serialize CRDT histories into metadata
  public serializeHistory(): string {
    return JSON.stringify({
      clocks: Array.from(this.peerClocks.entries()),
      history: this.historyStack,
    });
  }

  // 541. Forking a session
  public forkLocal(): IModelGraph | null {
    if (!this.modelRef) return null;
    const forked = JSON.parse(JSON.stringify(this.modelRef));
    forked.name += '_forked';

    // Detach from current CRDT updates
    return forked;
  }

  private syncToModel(): void {
    if (!this.modelRef) return;

    // Rebuild active node list from CRDT
    const activeNodes: INode[] = [];
    this.nodeMap.forEach((state) => {
      if (!state.deleted) {
        activeNodes.push(JSON.parse(JSON.stringify(state.node)));
      }
    });

    this.modelRef.nodes = activeNodes;
    globalEvents.emit('modelLoaded', this.modelRef); // Trigger re-render UI
  }
}
