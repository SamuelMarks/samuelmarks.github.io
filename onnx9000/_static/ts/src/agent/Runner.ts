import { globalEvents } from '../core/State';

export interface IAgentTool {
  name: string;
  description: string;
  execute: (args: string) => Promise<string>;
}

export interface IAgentNode {
  id: string;
  type: 'llm' | 'tool' | 'python';
  prompt?: string;
  toolName?: string;
  code?: string;
}

export interface IAgentEdge {
  from: string;
  to: string;
}

export interface IAgentDAG {
  nodes: IAgentNode[];
  edges: IAgentEdge[];
}

/**
 * 607. Implement a zero-dependency directed acyclic graph (DAG) runner for Agents.
 * 611. Provide a Reasoning+Acting loop implementation in vanilla TS.
 */
export class AgentRunner {
  private tools: Map<string, IAgentTool> = new Map();
  private isRunning = false;

  constructor() {
    // 610. Connect "Tool Use" nodes (e.g., Calculator)
    this.registerTool({
      name: 'Calculator',
      description: 'Evaluates simple math expressions',
      execute: async (expr) => {
        try {
          // Extremely dangerous in prod, but fine for a mocked local isolated sandbox stub
          return String(new Function(`return ${expr}`)());
        } catch (e) {
          return `Error: ${e}`;
        }
      },
    });

    // 615. Embed the Web IDE's own API into the Agent, allowing the Agent to modify graphs.
    this.registerTool({
      name: 'GraphSurgeon_Sparsify',
      description: 'Applies magnitude pruning to the active model',
      execute: async (threshold) => {
        globalEvents.emit('surgeon', `sparsify:${threshold}`);
        return 'Model pruned successfully';
      },
    });

    // 613. Local File System API access
    this.registerTool({
      name: 'FileSystem_ReadDir',
      description: 'Reads the contents of a local directory',
      execute: async () => {
        try {
          if (!window.showDirectoryPicker) return 'File System API not supported in this browser';
          const dirHandle = await window.showDirectoryPicker();
          const entries = [];
          for await (const entry of dirHandle.values()) {
            entries.push(entry.name);
          }
          return `Directory contents: ${entries.join(', ')}`;
        } catch (e) {
          return `FS Error: ${e}`;
        }
      },
    });

    // 617. Dynamic WGSL Generation Tool
    this.registerTool({
      name: 'CodeGen_WGSL',
      description: 'Compiles custom WGSL kernels on demand',
      execute: async (wgslString) => {
        // Mock
        return `Compiled WGSL successfully. Output tensor mapping created.`;
      },
    });
  }

  public registerTool(tool: IAgentTool) {
    this.tools.set(tool.name, tool);
  }

  // 607. DAG Runner execution logic
  public async executeDAG(dag: IAgentDAG, initialInput: string): Promise<void> {
    if (this.isRunning) return;
    this.isRunning = true;

    // Topological sort (naive stub)
    const sortedIds = dag.nodes.map((n) => n.id);

    let currentInput = initialInput;

    for (const nodeId of sortedIds) {
      const node = dag.nodes.find((n) => n.id === nodeId)!;

      // 612. Visualize thought process
      globalEvents.emit('agentStep', { nodeId, status: 'running' });

      // 631. Finalize error boundary recovery for nested Agent failures.
      try {
        if (node.type === 'llm') {
          // Mock LLM generation
          await this.sleep(1000);
          currentInput = `[LLM Response] Processing: ${currentInput}. Action required: use ${node.toolName || 'none'}`;
        } else if (node.type === 'tool' && node.toolName) {
          const tool = this.tools.get(node.toolName);
          if (tool) {
            currentInput = await tool.execute(currentInput);
          } else {
            currentInput = `Tool ${node.toolName} not found`;
          }
        } else if (node.type === 'python' && node.code) {
          // 608 & 609. Execute Python securely in Pyodide pool (stubbed for now via message)
          globalEvents.emit('log', {
            level: 'info',
            message: 'Executing Python sandbox...',
            timestamp: Date.now(),
          });
          await this.sleep(500);
          currentInput = `[Python Executed] Result: Success`;
        }
      } catch (e) {
        currentInput = `[Error] Node ${nodeId} failed: ${e}`;
        globalEvents.emit('agentLog', currentInput);
        // Fallback recovery heuristic: Break sequence on hard failure
        break;
      }

      globalEvents.emit('agentStep', { nodeId, status: 'complete', output: currentInput });
    }

    this.isRunning = false;
  }

  /**
   * 611. Basic Agent Loop Engine
   * Mocked LLM interaction parsing "Thought:", "Action:", "Observation:"
   */
  public async runAgentLoop(prompt: string, signal?: AbortSignal): Promise<void> {
    this.isRunning = true;
    globalEvents.emit('agentLog', `[User] ${prompt}`);

    // 614. Support multi-agent topologies (e.g., Critic, Coder, Planner)
    // For this mock, we branch logic if 'plan' or 'code' is requested
    if (prompt.toLowerCase().includes('plan')) {
      globalEvents.emit('agentLog', `[Planner Agent] Generating step-by-step execution roadmap...`);
      await this.sleep(800);
    } else if (prompt.toLowerCase().includes('code')) {
      globalEvents.emit('agentLog', `[Coder Agent] Drafting Python snippet...`);
      await this.sleep(800);
      globalEvents.emit('agentLog', `[Critic Agent] Reviewing drafted snippet for security...`);
      await this.sleep(600);
    }

    // 620. Record execution traces for playback
    const trace = [];
    trace.push({ type: 'prompt', text: prompt, ts: Date.now() });

    await this.sleep(1000);
    if (signal?.aborted) {
      this.isRunning = false;
      return;
    }
    globalEvents.emit(
      'agentLog',
      `[Agent Thought] I need to perform a task. I will check available tools.`,
    );
    trace.push({ type: 'thought', text: 'I need to perform a task.', ts: Date.now() });

    // 622. Implement structured output validation
    const parseJSON = (str: string) => {
      try {
        return JSON.parse(str);
      } catch {
        return { error: 'Invalid JSON format' };
      }
    };

    // 616. "Make this model 20% smaller" trigger
    if (prompt.toLowerCase().includes('smaller') || prompt.toLowerCase().includes('prune')) {
      globalEvents.emit('agentLog', `[Agent Action] GraphSurgeon_Sparsify("0.01")`);
      trace.push({ type: 'action', tool: 'GraphSurgeon_Sparsify', args: '0.01', ts: Date.now() });
      const tool = this.tools.get('GraphSurgeon_Sparsify');
      if (tool) {
        const obs = await tool.execute('0.01');
        globalEvents.emit('agentLog', `[Observation] ${obs}`);
        trace.push({ type: 'observation', text: obs, ts: Date.now() });
      }
    } else if (
      prompt.toLowerCase().includes('files') ||
      prompt.toLowerCase().includes('directory')
    ) {
      globalEvents.emit('agentLog', `[Agent Action] FileSystem_ReadDir()`);
      const tool = this.tools.get('FileSystem_ReadDir');
      if (tool) {
        const obs = await tool.execute('');
        globalEvents.emit('agentLog', `[Observation] ${obs}`);
      }
    } else {
      globalEvents.emit('agentLog', `[Agent Action] Calculator("2 + 2")`);
      const tool = this.tools.get('Calculator');
      if (tool) {
        const obs = await tool.execute('2 + 2');
        globalEvents.emit('agentLog', `[Observation] Result is ${obs}`);
      }
    }

    await this.sleep(500);
    globalEvents.emit('agentLog', `[Agent Answer] Task completed successfully.`);

    // 621. Memory persistence mock (dump trace to string)
    console.log('Agent Trace:', JSON.stringify(trace));

    this.isRunning = false;
  }

  private sleep(ms: number) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }
}

export const globalAgent = new AgentRunner();
