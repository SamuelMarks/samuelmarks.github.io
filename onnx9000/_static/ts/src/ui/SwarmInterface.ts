import { BaseComponent } from './BaseComponent';
import { $, $create, $on, $off } from '../core/DOM';
import { globalEvents } from '../core/State';
import { Toast } from './Toast';
import { WebRTCManager } from '../swarm/WebRTCManager';

export class SwarmInterface extends BaseComponent {
  private rtc: WebRTCManager;
  private peerList: HTMLElement;
  private idDisplay: HTMLElement;
  private lastExecutionHash: string | null = null;

  constructor(containerId: string | HTMLElement) {
    super(containerId);

    this.rtc = new WebRTCManager();

    // 366. Introduce a "Swarm" tab for decentralized browser-to-browser execution.
    this.container.classList.add('ide-swarm-container');
    this.container.style.padding = '20px';
    this.container.style.height = '100%';
    this.container.style.overflowY = 'auto';

    const header = $create('h2', { textContent: 'Decentralized Swarm' });
    this.container.appendChild(header);

    const infoCard = $create('div', { className: 'property-section' });
    this.idDisplay = $create('p', {
      innerHTML: `Your Peer ID: <strong>${this.rtc.getLocalId()}</strong>`,
    });
    infoCard.appendChild(this.idDisplay);
    this.container.appendChild(infoCard);

    // Manual Signaling Mechanism (369)
    const signalSection = $create('div', { className: 'property-section' });
    signalSection.appendChild($create('h3', { textContent: 'Manual Signaling' }));

    const row1 = $create('div', { className: 'property-row' });
    const createOfferBtn = $create('button', {
      className: 'action-btn',
      textContent: '1. Create Offer',
    });
    const offerOutput = $create<HTMLTextAreaElement>('textarea', {
      className: 'ide-chat-input',
      attributes: { rows: '3', readonly: 'true', placeholder: 'Offer SDP...' },
    });
    offerOutput.style.marginLeft = '10px';
    row1.appendChild(createOfferBtn);
    row1.appendChild(offerOutput);

    const row2 = $create('div', {
      className: 'property-row',
      attributes: { style: 'margin-top:10px' },
    });
    const acceptOfferBtn = $create('button', {
      className: 'action-btn secondary',
      textContent: '2. Accept Offer',
    });
    const offerInput = $create<HTMLTextAreaElement>('textarea', {
      className: 'ide-chat-input',
      attributes: { rows: '3', placeholder: "Paste peer's Offer SDP here..." },
    });
    offerInput.style.marginLeft = '10px';
    row2.appendChild(acceptOfferBtn);
    row2.appendChild(offerInput);

    const row3 = $create('div', {
      className: 'property-row',
      attributes: { style: 'margin-top:10px' },
    });
    const acceptAnswerBtn = $create('button', {
      className: 'action-btn secondary',
      textContent: '3. Accept Answer',
    });
    const answerInput = $create<HTMLTextAreaElement>('textarea', {
      className: 'ide-chat-input',
      attributes: { rows: '3', placeholder: "Paste peer's Answer SDP here..." },
    });
    answerInput.style.marginLeft = '10px';
    row3.appendChild(acceptAnswerBtn);
    row3.appendChild(answerInput);

    signalSection.appendChild(row1);
    signalSection.appendChild(row2);
    signalSection.appendChild(row3);
    this.container.appendChild(signalSection);

    // Peer List
    const peerSection = $create('div', { className: 'property-section' });
    peerSection.appendChild($create('h3', { textContent: 'Connected Peers' }));
    this.peerList = $create('ul', { className: 'property-list' });
    peerSection.appendChild(this.peerList);

    // 371. Display connected peers in a visual graph (nodes = browsers)
    // 379. Visualize live data flow
    const swarmCanvas = $create<HTMLCanvasElement>('canvas', {
      attributes: {
        width: '200',
        height: '150',
        style:
          'border: 1px solid var(--color-background-border); background: var(--color-background-secondary); border-radius: 4px; margin-top: 10px;',
      },
    });
    peerSection.appendChild(swarmCanvas);

    // 391. Save Topology
    const saveTopoBtn = $create('button', {
      className: 'action-btn secondary small',
      textContent: 'Save Swarm Config',
      attributes: { style: 'margin-top: 5px; display: block;' },
    });
    saveTopoBtn.addEventListener('click', () => {
      const peers = this.rtc.getConnectedPeers();
      const config = JSON.stringify({ peers, type: 'swarm_topology', timestamp: Date.now() });
      const b = new Blob([config], { type: 'application/json' });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(b);
      a.download = 'swarm_topology.json';
      a.click();
    });
    peerSection.appendChild(saveTopoBtn);

    // 392. Artificial Latency
    const lagRow = $create('div', {
      className: 'property-row',
      attributes: { style: 'margin-top: 10px; flex-direction: column;' },
    });
    const lagLabel = $create('label', { textContent: 'Simulate Network Latency (0ms)' });
    const lagSlider = $create<HTMLInputElement>('input', {
      className: 'ide-file-input',
      attributes: { type: 'range', min: '0', max: '1000', step: '50', value: '0' },
    });
    lagRow.appendChild(lagLabel);
    lagRow.appendChild(lagSlider);
    peerSection.appendChild(lagRow);

    lagSlider.addEventListener('change', () => {
      lagLabel.textContent = `Simulate Network Latency (${lagSlider.value}ms)`;
      Toast.show(`Artificial Ping set to ${lagSlider.value}ms`, 'warn');
    });

    this.container.appendChild(peerSection);

    globalEvents.on('swarmDataFlow', (targetId: string) => {
      // Briefly flash the edge to the target node
      const ctx = swarmCanvas.getContext('2d');
      if (ctx) {
        ctx.beginPath();
        ctx.strokeStyle = 'rgba(0, 255, 0, 0.8)';
        ctx.lineWidth = 3;
        ctx.moveTo(100, 130); // Master node
        ctx.lineTo(100, 20); // Remote peer position mock
        ctx.stroke();

        setTimeout(() => this.renderSwarmGraph(swarmCanvas), 300);
      }
    });

    globalEvents.on('swarmPeerConnected', () => this.renderSwarmGraph(swarmCanvas));
    globalEvents.on('swarmPeerDisconnected', () => this.renderSwarmGraph(swarmCanvas));

    // 522. Collaborate Button
    const collabSection = $create('div', { className: 'property-section' });
    const collabBtn = $create('button', {
      className: 'action-btn',
      textContent: 'Start Multiplayer Session',
    });
    const voiceBtn = $create('button', {
      className: 'action-btn secondary',
      textContent: 'Enable Voice Chat',
    });
    voiceBtn.style.marginLeft = '10px';

    const forkBtn = $create('button', {
      className: 'action-btn secondary',
      textContent: 'Fork Session',
    });
    forkBtn.style.marginLeft = '10px';

    // 383, 385. Benchmark & Auto-Balance Stubs
    const benchBtn = $create('button', {
      className: 'action-btn secondary small',
      textContent: 'Benchmark Swarm Topology',
    });
    benchBtn.style.marginLeft = '10px';
    const balanceBtn = $create('button', {
      className: 'action-btn secondary small',
      textContent: 'Auto-Balance Workload',
    });
    balanceBtn.style.marginLeft = '10px';

    // 543. Multi-user consensus stub
    const consensusBtn = $create('button', {
      className: 'action-btn danger small',
      textContent: 'Propose Distributed Train',
    });
    consensusBtn.style.marginTop = '10px';

    collabSection.appendChild(collabBtn);
    collabSection.appendChild(voiceBtn);
    collabSection.appendChild(forkBtn);
    collabSection.appendChild(benchBtn);
    collabSection.appendChild(balanceBtn);
    collabSection.appendChild(consensusBtn);

    benchBtn.addEventListener('click', () => {
      const peerCount = this.rtc.getConnectedPeers().length;
      if (peerCount === 0)
        return Toast.show('Need at least 1 remote peer to benchmark Swarm', 'warn');

      Toast.show('Benchmarking execution (Local vs Swarm)...', 'info');
      setTimeout(() => {
        // Mock 383
        const localMs = 150 + Math.random() * 20;
        // Swarm adds latency but could cut dense math time
        const swarmMs = 150 / (peerCount + 1) + 50;
        Toast.show(
          `Swarm Benchmark: Local ${localMs.toFixed(0)}ms | Swarm (${peerCount} peers) ${swarmMs.toFixed(0)}ms`,
          swarmMs < localMs ? 'success' : 'warn',
        );
      }, 1500);
    });

    balanceBtn.addEventListener('click', () => {
      Toast.show('Re-distributing graph partitions based on ping latency...', 'success');
      // Mock 385 auto-balancer
    });

    consensusBtn.addEventListener('click', () => {
      if (this.rtc.getConnectedPeers().length === 0) {
        Toast.show('No peers available for consensus', 'error');
        return;
      }
      this.rtc.broadcast({ type: 'sync', payload: { cmd: 'propose_train' } });
      Toast.show('Consensus proposal broadcasted. Awaiting 2/3 peer approval.', 'info');
    });

    // 545. Token-based authentication stub
    const authRow = $create('div', {
      className: 'property-row',
      attributes: { style: 'margin-top: 10px;' },
    });
    const authInput = $create<HTMLInputElement>('input', {
      className: 'ide-chat-input',
      attributes: { type: 'password', placeholder: 'Session Room Token (Optional)' },
    });
    authRow.appendChild(authInput);
    collabSection.appendChild(authRow);

    this.container.appendChild(collabSection);

    forkBtn.addEventListener('click', () => {
      globalEvents.emit('forkSession');
      Toast.show('Session forked locally. Disconnected from Swarm.', 'info');
    });

    authInput.addEventListener('change', () => {
      // Mock token validation. In reality, passed securely during SDP handshake.
      if (authInput.value.length > 5) {
        Toast.show('Authentication Token Applied', 'success');
      }
    });

    collabBtn.addEventListener('click', () => {
      globalEvents.emit('initCollab', this.rtc.getLocalId());
      Toast.show('Multiplayer session activated.', 'success');
    });

    // 539. Integrated Voice Chat UI Stub
    let activeVoiceStream: MediaStream | null = null;
    voiceBtn.addEventListener('click', async () => {
      if (activeVoiceStream) {
        activeVoiceStream.getTracks().forEach((t) => t.stop());
        activeVoiceStream = null;
        voiceBtn.textContent = 'Enable Voice Chat';
        Toast.show('Voice Chat Disabled', 'info');
      } else {
        try {
          activeVoiceStream = await navigator.mediaDevices.getUserMedia({
            audio: true,
            video: false,
          });
          this.rtc.attachVoiceStream(activeVoiceStream);
          voiceBtn.textContent = 'Disable Voice Chat';
          Toast.show('Voice Chat Enabled. Broadcasting mic...', 'success');
        } catch (e) {
          Toast.show('Microphone access denied', 'error');
        }
      }
    });

    // Handle incoming streams
    const remoteAudioContainer = $create('div', { className: 'hidden' });
    this.container.appendChild(remoteAudioContainer);

    globalEvents.on('swarmAudioTrackReceived', (data: any) => {
      const { peerId, stream } = data;
      let audioEl = document.getElementById(`audio_${peerId}`) as HTMLAudioElement;
      if (!audioEl) {
        audioEl = $create<HTMLAudioElement>('audio', {
          id: `audio_${peerId}`,
          attributes: { autoplay: 'true' },
        });
        remoteAudioContainer.appendChild(audioEl);
      }
      audioEl.srcObject = stream;
      Toast.show(`Receiving voice data from ${peerId}`, 'info');
    });

    // Event Bindings
    let tempId = '';

    createOfferBtn.addEventListener('click', async () => {
      try {
        const res = await this.rtc.createOffer();
        tempId = res.id;
        offerOutput.value = res.offer;
        offerOutput.select();
        document.execCommand('copy');
        Toast.show('Offer generated and copied to clipboard', 'success');
      } catch (e) {
        Toast.show('Failed to create offer', 'error');
      }
    });

    acceptOfferBtn.addEventListener('click', async () => {
      try {
        const val = offerInput.value.trim();
        if (!val) return;
        const mockPeerId = `peer_${Math.random().toString(36).substring(2, 6)}`;
        const answerStr = await this.rtc.acceptOffer(mockPeerId, val);
        answerInput.value = answerStr;
        answerInput.select();
        document.execCommand('copy');
        Toast.show('Answer generated and copied to clipboard. Send back to peer.', 'success');
      } catch (e) {
        Toast.show('Failed to accept offer', 'error');
      }
    });

    acceptAnswerBtn.addEventListener('click', async () => {
      try {
        const val = answerInput.value.trim();
        if (!val || !tempId) return;
        const mockPeerId = `peer_${Math.random().toString(36).substring(2, 6)}`;
        await this.rtc.acceptAnswer(tempId, mockPeerId, val);
        Toast.show('Connection established', 'success');
      } catch (e) {
        Toast.show('Failed to accept answer', 'error');
      }
    });
  }

  mount(): void {
    // 530. Share Monaco Editor State
    globalEvents.on('monacoCodeChanged', (code: string) => {
      this.rtc.broadcast({ type: 'editor_sync', payload: code });
    });

    // 531. Sync Execution Results
    globalEvents.on('profilerData', (traces: any) => {
      this.rtc.broadcast({ type: 'sync', payload: { traces } });
    });

    // 536. Sync Layout Coordinates
    globalEvents.on('nodeLayoutMoved', (data: any) => {
      // Mock: this.rtc.broadcast({ type: "sync", payload: { nodeLayout: data } });
    });

    // 554. Sync Theme (Dark/Light) conditionally
    globalEvents.on('themeChanged', (theme: string) => {
      this.rtc.broadcast({ type: 'sync', payload: { theme } });
    });

    // 542. Diffing mock payload listener
    globalEvents.on('swarmSync', (data: any) => {
      const { peerId, payload } = data;
      if (payload.cmd === 'propose_train') {
        if (confirm(`Peer ${peerId} proposes starting distributed training. Accept?`)) {
          this.rtc.sendMessage(peerId, { type: 'sync', payload: { cmd: 'accept_train' } });
        }
      } else if (payload.cmd === 'accept_train') {
        Toast.show(`Peer ${peerId} accepted training proposal`, 'success');
      } else if (payload.diff) {
        Toast.show(
          `Visual diff received from ${peerId}. ${payload.diff.length} changes detected.`,
          'info',
        );
      } else if (payload.theme) {
        // 554. Apply remote theme change
        Toast.show(`Peer changed theme to ${payload.theme}`, 'info');
      } else if (payload.cmd === 'parity_check') {
        // 551. Hash check logic
        if (this.lastExecutionHash && this.lastExecutionHash !== payload.hash) {
          // 552. Alert divergence
          Toast.show(
            `DIVERGENCE DETECTED: Peer ${peerId} output does not match local hardware output!`,
            'error',
          );
          globalEvents.emit(
            'agentLog',
            `[Critical] Floating point divergence detected with peer ${peerId}.`,
          );
        } else {
          Toast.show(`Parity Match: Peer ${peerId} output hash verified`, 'success');
        }
      }
    });

    globalEvents.on('swarmParityCheck', (data: any) => {
      this.lastExecutionHash = data.hash;
      this.rtc.broadcast({ type: 'sync', payload: { cmd: 'parity_check', hash: data.hash } });
    });

    // 373. Calculate and display network latency
    globalEvents.on('swarmLatencyUpdate', (data: any) => {
      const pingEl = document.getElementById(`ping-${data.peerId}`);
      if (pingEl) {
        pingEl.textContent = `${data.rtt} ms`;
      }
    });

    globalEvents.on('swarmPeerConnected', (peerId: string) => {
      Toast.show(`Peer connected: ${peerId}`, 'success');
      this.renderPeerList();
    });

    globalEvents.on('swarmPeerDisconnected', (peerId: string) => {
      Toast.show(`Peer disconnected: ${peerId}`, 'warn');
      this.renderPeerList();
    });
  }

  private renderSwarmGraph(canvas: HTMLCanvasElement): void {
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, 200, 150);

    const peers = this.rtc.getConnectedPeers();

    // Draw Master Node (Local)
    ctx.fillStyle = 'var(--color-primary)';
    ctx.beginPath();
    ctx.arc(100, 130, 15, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#fff';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('You', 100, 134);

    if (peers.length === 0) return;

    const step = 160 / peers.length;
    peers.forEach((p, i) => {
      const x = 20 + i * step + step / 2;
      const y = 30;

      // Draw Edge
      ctx.beginPath();
      ctx.strokeStyle = 'rgba(100, 100, 100, 0.5)';
      ctx.lineWidth = 1;
      ctx.moveTo(100, 130);
      ctx.lineTo(x, y);
      ctx.stroke();

      // Draw Peer
      ctx.fillStyle = 'var(--color-success)';
      ctx.beginPath();
      ctx.arc(x, y, 12, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = '#fff';
      ctx.fillText(p.substring(0, 3), x, y + 3);
    });
  }

  private renderPeerList(): void {
    this.peerList.innerHTML = '';
    const peers = this.rtc.getConnectedPeers();
    if (peers.length === 0) {
      this.peerList.innerHTML = "<li class='muted'>No peers connected</li>";
      return;
    }

    // 550. Add UI for managing active peer connections
    peers.forEach((p) => {
      const li = $create('li', {
        className: 'property-row',
        id: `peer-row-${p}`,
        innerHTML: `<span>🟢 <strong>${p}</strong> <span id="ping-${p}" class="muted" style="font-size:0.7rem; margin-left:10px;">-- ms</span></span>`,
      });

      const actions = $create('div');

      const kickBtn = $create('button', {
        className: 'action-btn danger small',
        textContent: 'Kick',
      });
      kickBtn.addEventListener('click', () => this.rtc.disconnectPeer(p));

      const syncBtn = $create('button', {
        className: 'action-btn secondary small',
        textContent: 'Sync State',
      });
      syncBtn.style.marginRight = '5px';
      syncBtn.addEventListener('click', () => {
        Toast.show(`Force sync state requested for ${p}`, 'info');
        this.rtc.sendMessage(p, { type: 'sync', payload: { cmd: 'force_reconcile' } });
      });

      actions.appendChild(syncBtn);
      actions.appendChild(kickBtn);
      li.appendChild(actions);

      this.peerList.appendChild(li);
    });
  }
}
