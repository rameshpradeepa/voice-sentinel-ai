/**
 * VoiceSentinel AI - Telephony & Security Scenario Simulator
 * Simulates high-pressure telephonic wire transfer attack scenarios.
 * Evaluates real audio files through the PyTorch backend (/api/v1/analyze-file).
 */

class CallSimulator {
  constructor(appController) {
    this.app = appController;
    this.activeCall = null;
  }

  async startCallSimulation(scenarioType = 'cxo_wire_transfer_attack') {
    if (this.activeCall) {
      this.endCall();
    }

    const scenarios = {
      cxo_wire_transfer_attack: {
        id: 'sim_' + Math.random().toString(36).substr(2, 6),
        callerName: 'Rajesh Sharma (Spoofed)',
        callerRole: 'Chief Financial Officer',
        callerNumber: '+91 98765 43210',
        targetProfileId: 'cxo_001',
        context: 'Urgent Wire Transfer Approval ($500,000 to Offshore Account)',
        samplePath: 'datasets/synthetic/default/sample_default_01.wav',
        isSpoofedCallerID: true,
        highRiskTransaction: true,
        voipProxyDetected: true,
        language: 'Indian English'
      },
      genuine_briefing_call: {
        id: 'sim_' + Math.random().toString(36).substr(2, 6),
        callerName: 'Dr. Anita Desai',
        callerRole: 'Director (Cyber Security)',
        callerNumber: '+91 98111 22334',
        targetProfileId: 'govt_002',
        context: 'Routine Departmental Briefing Call',
        samplePath: 'datasets/genuine/default/sample_default_01.wav',
        isSpoofedCallerID: false,
        highRiskTransaction: false,
        voipProxyDetected: false,
        language: 'Hindi / English Hybrid'
      }
    };

    this.activeCall = scenarios[scenarioType] || scenarios.cxo_wire_transfer_attack;
    this.updateCallUI('CONNECTING');

    setTimeout(async () => {
      this.updateCallUI('CONNECTED');
      // Fetch real sample file and pass to FastAPI backend for PyTorch inference
      try {
        const fileRes = await fetch(this.activeCall.samplePath);
        if (fileRes.ok) {
          const blob = await fileRes.blob();
          const file = new File([blob], 'simulator_sample.wav', { type: 'audio/wav' });
          const analysisRes = await this.app.apiClient.analyzeAudioFile(file, this.activeCall.targetProfileId);
          if (analysisRes.success && analysisRes.data) {
            this.app.renderAnalysisResult(analysisRes.data);
          }
        } else {
          // Fallback feature analysis if sample file path not fetched
          const analysisRes = await this.app.apiClient.analyzeStreamPayload({
            acousticFeatures: { pitchMean: 144.2, pitchStdDev: 3.1, highFreqEnergyRatio: 0.015 },
            callMetadata: {
              isSpoofedCallerID: this.activeCall.isSpoofedCallerID,
              highRiskTransaction: this.activeCall.highRiskTransaction,
              voipProxyDetected: this.activeCall.voipProxyDetected
            },
            targetProfileId: this.activeCall.targetProfileId
          });
          if (analysisRes.success && analysisRes.data) {
            this.app.renderAnalysisResult(analysisRes.data);
          }
        }
      } catch (err) {
        console.warn("Simulator backend fetch error:", err);
      }
    }, 1200);
  }

  endCall() {
    if (this.activeCall) {
      this.updateCallUI('ENDED');
      this.activeCall = null;
    }
  }

  updateCallUI(status) {
    const statusLabel = document.getElementById('sim-call-status');
    const callerName = document.getElementById('sim-caller-name');
    const callerRole = document.getElementById('sim-caller-role');
    const callerNum = document.getElementById('sim-caller-num');
    const contextBox = document.getElementById('sim-call-context');

    if (statusLabel) statusLabel.innerText = status;
    if (this.activeCall) {
      if (callerName) callerName.innerText = this.activeCall.callerName;
      if (callerRole) callerRole.innerText = this.activeCall.callerRole;
      if (callerNum) callerNum.innerText = this.activeCall.callerNumber;
      if (contextBox) contextBox.innerText = this.activeCall.context;
    } else {
      if (statusLabel) statusLabel.innerText = 'IDLE';
    }
  }
}

window.CallSimulator = CallSimulator;
