/**
 * VoiceSentinel AI - Authoritative Application Controller (v3.0)
 * Orchestrates Real-Time WebSocket Audio Analysis, File Upload Forensics,
 * PyTorch Model Results, Dynamic Risk Formula, Security Action Panel, and On-Chain Audit Logging.
 */

document.addEventListener('DOMContentLoaded', () => {
  window.appController = new AppController();
});

class AppController {
  constructor() {
    this.audioEngine = window.voiceAudioEngine;
    this.spectrogramRenderer = new window.SpectrogramRenderer('spectrogram-canvas', 'oscilloscope-canvas');
    this.callSimulator = new window.CallSimulator(this);
    this.apiClient = new window.ApiClient();

    this.currentAnalysisData = null;
    this.callDecisionState = "IDLE"; // ALLOW, VERIFY, REPORTED, BLOCKED, IDLE

    this.initUI();
    this.fetchModelMetadata();
    this.loadDiskDatasets();
    this.fetchLiveStats();
  }

  initUI() {
    // 1. Tab Navigation
    const tabBtns = document.querySelectorAll('.tab-btn');
    tabBtns.forEach(btn => {
      btn.addEventListener('click', (e) => {
        const tabTarget = btn.getAttribute('data-tab');
        
        tabBtns.forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));

        btn.classList.add('active');
        const activePane = document.getElementById(tabTarget);
        if (activePane) activePane.classList.add('active');
      });
    });

    // 2. Microphone Stream Controls & WebSocket Status
    const startMicBtn = document.getElementById('btn-start-mic');
    const stopMicBtn = document.getElementById('btn-stop-mic');
    const wsStatusLabel = document.getElementById('ws-status-label');

    if (startMicBtn) {
      startMicBtn.addEventListener('click', async () => {
        startMicBtn.disabled = true;
        if (stopMicBtn) stopMicBtn.disabled = false;

        const success = await this.audioEngine.startMicrophone(
          (features, freqData, timeData) => this.renderDspFrame(features, freqData, timeData),
          (pytorchResult) => this.renderPyTorchResult(pytorchResult),
          (isConnected, errorMsg) => {
            if (wsStatusLabel) {
              if (isConnected) {
                wsStatusLabel.innerText = "STREAMING TO PYTORCH MODEL";
                wsStatusLabel.style.color = "var(--status-low)";
              } else {
                wsStatusLabel.innerText = errorMsg || "REAL-TIME MODEL CONNECTION LOST";
                wsStatusLabel.style.color = "var(--status-critical)";
              }
            }
          }
        );
      });
    }

    if (stopMicBtn) {
      stopMicBtn.addEventListener('click', () => {
        this.audioEngine.stopMicrophone();
        startMicBtn.disabled = false;
        stopMicBtn.disabled = true;
        if (wsStatusLabel) {
          wsStatusLabel.innerText = "IDLE / STOPPED";
          wsStatusLabel.style.color = "var(--text-muted)";
        }
      });
    }

    // 3. Audio File Upload Form Inspector
    const fileInput = document.getElementById('audio-file-input');
    const btnUpload = document.getElementById('btn-analyze-file');
    const selectionCard = document.getElementById('file-selection-card');
    const selName = document.getElementById('sel-file-name');
    const selFormat = document.getElementById('sel-file-format');
    const selSize = document.getElementById('sel-file-size');
    const selStatus = document.getElementById('sel-file-status');
    const trackerBox = document.getElementById('upload-pipeline-tracker');
    const trackerText = document.getElementById('pipeline-status-text');

    if (fileInput) {
      fileInput.addEventListener('change', () => {
        if (fileInput.files && fileInput.files.length > 0) {
          const f = fileInput.files[0];
          const ext = f.name.includes('.') ? f.name.split('.').pop().toUpperCase() : 'UNKNOWN';
          const sizeMb = (f.size / (1024 * 1024)).toFixed(2);
          
          if (selName) selName.innerText = f.name;
          if (selFormat) selFormat.innerText = ext;
          if (selSize) selSize.innerText = `${sizeMb} MB (${f.size.toLocaleString()} bytes)`;
          if (selStatus) {
            selStatus.innerText = "READY";
            selStatus.className = "verdict-badge verdict-low";
          }
          if (selectionCard) selectionCard.style.display = 'block';
          if (trackerBox) trackerBox.style.display = 'none';
        } else {
          if (selectionCard) selectionCard.style.display = 'none';
        }
      });
    }

    if (btnUpload && fileInput) {
      btnUpload.addEventListener('click', async () => {
        if (!fileInput.files || fileInput.files.length === 0) {
          alert("Please select an audio file (WAV, FLAC, OGG, MP3, AAC) to inspect.");
          return;
        }
        const file = fileInput.files[0];
        btnUpload.disabled = true;
        btnUpload.innerText = "⏳ Processing Audio...";

        if (trackerBox && trackerText) {
          trackerBox.style.display = 'block';
          trackerText.innerText = "UPLOADING";
          trackerText.style.color = "var(--accent-cyan)";
        }

        const t1 = setTimeout(() => {
          if (trackerText && btnUpload.disabled) {
            trackerText.innerText = "DECODING WITH FFMPEG";
          }
        }, 200);

        const t2 = setTimeout(() => {
          if (trackerText && btnUpload.disabled) {
            trackerText.innerText = "ANALYZING WITH PYTORCH";
          }
        }, 600);

        const result = await this.apiClient.analyzeAudioFile(file);
        clearTimeout(t1);
        clearTimeout(t2);

        btnUpload.disabled = false;
        btnUpload.innerText = "🔍 Analyze Audio File";

        if (result.success && result.data) {
          if (trackerText) {
            trackerText.innerText = "COMPLETE";
            trackerText.style.color = "var(--status-low)";
          }
          if (selStatus) {
            selStatus.innerText = "INSPECTED";
            selStatus.className = "verdict-badge verdict-low";
          }
          this.renderAnalysisResult(result.data);
        } else {
          const errorMsg = result.message || result.detail || result.error || "Unable to decode the uploaded audio file.";
          if (trackerText) {
            trackerText.innerText = `DECODE FAILED: ${errorMsg}`;
            trackerText.style.color = "var(--status-critical)";
          }
          if (selStatus) {
            selStatus.innerText = "DECODE FAILED";
            selStatus.className = "verdict-badge verdict-critical";
          }
          alert(`Analysis Error:\n${errorMsg}`);
        }
      });
    }

    // 4. Scenario Simulator Selectors
    const simCxoAttackBtn = document.getElementById('btn-sim-cxo-attack');
    const simGenuineBtn = document.getElementById('btn-sim-genuine');
    const simEndBtn = document.getElementById('btn-sim-end');

    if (simCxoAttackBtn) simCxoAttackBtn.addEventListener('click', () => this.callSimulator.startCallSimulation('cxo_wire_transfer_attack'));
    if (simGenuineBtn) simGenuineBtn.addEventListener('click', () => this.callSimulator.startCallSimulation('genuine_briefing_call'));
    if (simEndBtn) simEndBtn.addEventListener('click', () => this.callSimulator.endCall());

    // 5. Interactive Security Actions Panel
    document.querySelectorAll('.btn-security-action').forEach(btn => {
      btn.addEventListener('click', async (e) => {
        const actionType = btn.getAttribute('data-action');
        await this.handleSecurityAction(actionType);
      });
    });

    // 6. Enroll Voice Profile Form
    const enrollForm = document.getElementById('form-enroll-voice');
    if (enrollForm) {
      enrollForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const name = document.getElementById('enroll-name').value;
        const org = document.getElementById('enroll-org').value;
        const pitch = parseFloat(document.getElementById('enroll-pitch').value) || 150;

        await fetch('/api/v1/enroll', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ name, organization: org, f0Mean: pitch })
        });
        alert(`✅ Acoustic Biometric Profile for ${name} enrolled successfully!`);
        enrollForm.reset();
        this.loadEnrolledProfiles();
      });
    }

    // 7. SIP Filter Test Form
    const sipForm = document.getElementById('form-sip-test');
    if (sipForm) {
      sipForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const header = document.getElementById('sip-header-input').value;
        const callerNum = document.getElementById('sip-caller-input').value;
        const result = await this.apiClient.testSipFilter(header, callerNum);
        document.getElementById('sip-result-box').innerText = JSON.stringify(result, null, 2);
      });
    }
  }

  renderDspFrame(features, freqData, timeData) {
    this.spectrogramRenderer.renderFrame(freqData, timeData);

    const pitchVal = document.getElementById('telemetry-pitch');
    const pitchStd = document.getElementById('telemetry-pitch-std');
    const centroidVal = document.getElementById('telemetry-centroid');
    const zcrVal = document.getElementById('telemetry-zcr');

    if (pitchVal) pitchVal.innerText = features.pitchHz ? `${features.pitchHz} Hz` : '--';
    if (pitchStd) pitchStd.innerText = `${features.pitchStdDev} Hz`;
    if (centroidVal) centroidVal.innerText = `${features.spectralCentroid} Hz`;
    if (zcrVal) zcrVal.innerText = features.zcr;
  }

  renderPyTorchResult(data) {
    this.currentAnalysisData = data;
    const aiProbPercent = data.aiProbabilityPercent || (data.aiProbability * 100);
    const humanProbPercent = data.humanProbabilityPercent || (100 - aiProbPercent);
    const riskResult = data.riskResult || {};
    const riskScore = riskResult.overallRiskScore || aiProbPercent;

    // AI & Human Probability Display
    const aiProbEl = document.getElementById('val-ai-prob');
    const humanProbEl = document.getElementById('val-human-prob');
    const aiBar = document.getElementById('bar-ai-prob');
    const humanBar = document.getElementById('bar-human-prob');

    if (aiProbEl) aiProbEl.innerText = `${aiProbPercent.toFixed(1)}%`;
    if (humanProbEl) humanProbEl.innerText = `${humanProbPercent.toFixed(1)}%`;
    if (aiBar) aiBar.style.width = `${aiProbPercent}%`;
    if (humanBar) humanBar.style.width = `${humanProbPercent}%`;

    // Risk Gauge & Verdict
    const scoreNumber = document.getElementById('risk-score-num');
    const progressRing = document.getElementById('risk-gauge-ring');
    const verdictBadge = document.getElementById('verdict-badge-box');

    if (scoreNumber) scoreNumber.innerText = `${riskScore.toFixed(1)}%`;
    if (progressRing) {
      const offset = 440 - (440 * (riskScore / 100));
      progressRing.style.strokeDashoffset = offset;
      progressRing.style.stroke = riskScore >= 80 ? 'var(--status-critical)' : (riskScore >= 60 ? 'var(--status-high)' : (riskScore >= 30 ? 'var(--status-medium)' : 'var(--status-low)'));
    }

    if (verdictBadge) {
      const alertLvl = riskResult.riskLevel || (riskScore >= 80 ? 'CRITICAL' : 'LOW');
      verdictBadge.innerText = riskResult.verdict || data.pyTorchVerdict;
      verdictBadge.className = `verdict-badge verdict-${alertLvl.toLowerCase()}`;
    }

    // Why Detected Explanations
    const whyBox = document.getElementById('why-detected-list');
    if (whyBox) {
      whyBox.innerHTML = '';
      const list = riskResult.whyDetected || ["PyTorch Model forward pass complete"];
      list.forEach(item => {
        const li = document.createElement('li');
        li.innerText = item;
        whyBox.appendChild(li);
      });
    }

    // Weighted Formula Breakdown
    const sub = riskResult.riskFormulaBreakdown || {};
    this.updateBar('bar-spectral', 'val-spectral', sub.aiModelContribution || (0.55 * aiProbPercent));
    this.updateBar('bar-prosody', 'val-prosody', sub.prosodyContribution || 10.0);
    this.updateBar('bar-biometric', 'val-biometric', sub.biometricContribution || 5.0);
    this.updateBar('bar-metadata', 'val-metadata', sub.metadataContribution || 0.0);

    // Technical Parameters Breakdown
    const tech = riskResult.technicalParameters || {};
    const spec = tech.spectral || {};
    const pros = tech.prosody || {};
    const qual = tech.voiceQuality || {};
    const bio = tech.biometric || {};

    const meta = data.audioMetadata || {};
    if (meta.durationSeconds !== undefined) {
      this.setText('tech-dur', `${meta.durationSeconds.toFixed(2)}s`);
      this.setText('tech-chunks', `${meta.chunkCount || 1} x 3s window(s)`);
      this.setText('tech-sr', `${meta.sampleRateHz || 16000} Hz`);
      this.setText('tech-channels', `${meta.channels || 'Mono'} (${meta.fileFormat || 'AUDIO'})`);
    } else if (data.chunksAnalyzed !== undefined) {
      this.setText('tech-dur', `${(data.chunksAnalyzed * 3).toFixed(1)}s`);
      this.setText('tech-chunks', `${data.chunksAnalyzed} window(s)`);
      this.setText('tech-sr', '16000 Hz');
      this.setText('tech-channels', 'Mono (Stream)');
    }

    this.setText('tech-centroid', spec.spectralCentroid ? `${spec.spectralCentroid} Hz` : '--');
    this.setText('tech-highfreq', spec.highFreqEnergyRatioPercent !== undefined ? `${spec.highFreqEnergyRatioPercent}%` : '--');
    this.setText('tech-zcr', spec.zcr !== undefined ? `${spec.zcr}` : '--');
    this.setText('tech-rms', spec.rmsEnergy !== undefined ? `${spec.rmsEnergy}` : (qual.rmsEnergy !== undefined ? `${qual.rmsEnergy}` : '--'));

    this.setText('tech-f0-mean', pros.f0MeanHz ? `${pros.f0MeanHz} Hz` : '--');
    this.setText('tech-f0-std', pros.f0StdDevHz ? `${pros.f0StdDevHz} Hz` : '--');
    this.setText('tech-pitch-range', pros.pitchRangeHz ? `${pros.pitchRangeHz} Hz` : '--');
    this.setText('tech-pause-ratio', pros.pauseRatio !== undefined ? `${(pros.pauseRatio * 100).toFixed(1)}%` : '--');

    this.setText('tech-bio-match', bio.matchPercentage !== undefined ? `${bio.matchPercentage}%` : '--');
    this.setText('tech-latency', data.inferenceLatencyMs !== undefined ? `${data.inferenceLatencyMs} ms` : '--');

    // SHA-256 & Audit Status
    const hashEl = document.getElementById('live-audio-hash');
    if (hashEl && data.audioHashSha256) hashEl.innerText = data.audioHashSha256;

    // Show action buttons matching risk level
    this.updateSecurityActionButtons(riskResult.riskLevel || 'LOW');
  }

  renderAnalysisResult(fileData) {
    const inf = fileData.pyTorchInference || {};
    const sec = fileData.securityRisk || {};

    const inspectorTab = document.querySelector('.tab-btn[data-tab="pane-inspector"]');
    if (inspectorTab) inspectorTab.click();

    this.renderPyTorchResult({
      aiProbability: inf.aiProbability,
      aiProbabilityPercent: inf.aiProbabilityPercent,
      humanProbabilityPercent: 100 - inf.aiProbabilityPercent,
      pyTorchVerdict: inf.verdict,
      audioHashSha256: fileData.audioHashSha256,
      riskResult: sec,
      audioMetadata: {
        fileName: fileData.fileName,
        fileFormat: fileData.fileFormat,
        fileSizeBytes: fileData.fileSizeBytes,
        durationSeconds: fileData.audioDurationSec,
        sampleRateHz: fileData.sampleRateHz,
        channels: fileData.channels,
        chunkCount: inf.chunksAnalyzed || 1
      },
      inferenceLatencyMs: inf.inferenceLatencyMs
    });
  }

  updateSecurityActionButtons(riskLevel) {
    const actionPanel = document.getElementById('security-actions-panel');
    if (!actionPanel) return;

    actionPanel.querySelectorAll('.btn-security-action').forEach(btn => btn.style.display = 'none');

    if (riskLevel === 'CRITICAL') {
      this.showBtn('act-block-call');
      this.showBtn('act-block-tx');
      this.showBtn('act-report');
      this.showBtn('act-mfa');
      this.showBtn('act-soc');
    } else if (riskLevel === 'HIGH') {
      this.showBtn('act-mfa');
      this.showBtn('act-verify');
      this.showBtn('act-report');
      this.showBtn('act-block-call');
    } else if (riskLevel === 'MEDIUM') {
      this.showBtn('act-verify');
      this.showBtn('act-challenge');
      this.showBtn('act-report');
    } else {
      this.showBtn('act-allow');
      this.showBtn('act-report');
    }
  }

  showBtn(btnId) {
    const btn = document.getElementById(btnId);
    if (btn) btn.style.display = 'inline-flex';
  }

  async handleSecurityAction(actionType) {
    const incId = "inc_" + Date.now();
    const riskScore = this.currentAnalysisData ? (this.currentAnalysisData.riskResult ? this.currentAnalysisData.riskResult.overallRiskScore : 0.0) : 0.0;
    const verdict = this.currentAnalysisData ? (this.currentAnalysisData.riskResult ? this.currentAnalysisData.riskResult.verdict : "UNINSPECTED") : "UNINSPECTED";
    const audioHash = this.currentAnalysisData ? this.currentAnalysisData.audioHashSha256 : "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

    const res = await this.apiClient.triggerSecurityAction(incId, actionType, riskScore, verdict, audioHash);
    const callStatusBadge = document.getElementById('decision-state-badge');
    const decisionNotice = document.getElementById('decision-state-notice');

    if (res.success && res.callDecisionState) {
      this.callDecisionState = res.callDecisionState;
      if (callStatusBadge) {
        callStatusBadge.innerText = `CALL STATUS: ${res.callDecisionState}`;
        callStatusBadge.className = `verdict-badge verdict-${res.callDecisionState === 'BLOCKED' ? 'critical' : (res.callDecisionState === 'ALLOW' ? 'low' : 'medium')}`;
      }

      if (decisionNotice) {
        if (res.callDecisionState === 'BLOCKED') {
          decisionNotice.innerHTML = `🛑 <strong>CALL BLOCKED</strong> — TRANSACTION HOLD REQUESTED. Security Operations notified.`;
        } else if (res.callDecisionState === 'ALLOW') {
          decisionNotice.innerHTML = `✅ <strong>CALL ALLOWED</strong> — Proceed with normal call flow.`;
        } else {
          decisionNotice.innerHTML = `⚠️ <strong>SECURITY ACTION EXECUTED (${actionType})</strong> — Incident ID ${incId} logged.`;
        }
      }

      const chainBadge = document.getElementById('blockchain-status-badge');
      if (chainBadge && res.incidentRecord && res.incidentRecord.blockchainAudit) {
        const bData = res.incidentRecord.blockchainAudit;
        if (bData.blockchainStatus === 'VERIFIED') {
          chainBadge.innerHTML = `<span>Blockchain: <strong style="color:var(--status-low);">VERIFIED (${bData.transactionHash.substr(0, 10)}...)</strong></span>`;
        } else {
          chainBadge.innerHTML = `<span>Blockchain: <strong style="color:var(--status-medium);">OFFLINE — SHA-256 audit recorded locally</strong></span>`;
        }
      }
    }
  }

  updateBar(barId, valId, score) {
    const bar = document.getElementById(barId);
    const val = document.getElementById(valId);
    if (bar) bar.style.width = `${Math.min(100, Math.max(0, score))}%`;
    if (val) val.innerText = `${score.toFixed(0)}%`;
  }

  setText(id, text) {
    const el = document.getElementById(id);
    if (el) el.innerText = text;
  }

  async fetchModelMetadata() {
    const res = await this.apiClient.getModelInfo();
    const modelBox = document.getElementById('model-info-box');
    if (!modelBox) return;

    if (res.success && res.modelMetadata) {
      const meta = res.modelMetadata;
      const evalData = res.evaluationMetrics;

      const evalHtml = (typeof evalData === 'object' && evalData.accuracy) ? `
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.4rem; font-family:var(--font-mono); margin-top:0.2rem;">
          <div>Accuracy: <span style="color:var(--status-low);">${evalData.accuracy}%</span></div>
          <div>F1 Score: <span style="color:var(--status-low);">${evalData.f1Score}%</span></div>
          <div>Precision: <span>${evalData.precision}%</span></div>
          <div>Recall: <span>${evalData.recall}%</span></div>
        </div>
      ` : `<div style="color:var(--status-medium); font-weight:700;">NOT EVALUATED</div>`;

      modelBox.innerHTML = `
        <div style="font-size:0.85rem; display:flex; flex-direction:column; gap:0.4rem;">
          <div>Architecture: <strong style="color:var(--accent-cyan);">${meta.modelName} (${meta.framework})</strong></div>
          <div>Status: <span class="verdict-badge verdict-low" style="padding:0.1rem 0.5rem; font-size:0.7rem;">${meta.status}</span></div>
          <div>Sample Rate: <strong>${meta.sampleRateHz} Hz (Log-Mel Spectrogram 64-bins)</strong></div>
          <div>Parameters: <strong>${meta.totalParameters.toLocaleString()}</strong></div>
          <div>Trained Samples: <strong>${meta.datasetSamples}</strong></div>
          <div style="border-top:1px dashed var(--border-color); padding-top:0.4rem; margin-top:0.2rem;">
            <strong>Evaluation Status & Test Metrics:</strong>
            ${evalHtml}
          </div>
        </div>
      `;
    } else {
      modelBox.innerHTML = `
        <div style="font-size:0.82rem; color:var(--status-medium);">
          ⚠️ NOT EVALUATED: PyTorch model weights not found.<br>
          Run dataset validator & training script: <code style="color:var(--accent-cyan);">python ml_engine/train.py</code>
        </div>
      `;
    }
  }

  async loadDiskDatasets() {
    const res = await this.apiClient.getDatasets();
    if (res.success && res.data) {
      this.renderDatasetsTable(res.data);
    }
  }

  renderDatasetsTable(datasetData) {
    const tableBody = document.getElementById('dataset-table-body');
    const diskSummary = document.getElementById('disk-dataset-summary');
    if (!tableBody) return;

    if (diskSummary) {
      diskSummary.innerHTML = `
        Disk Audio Files: <strong>${datasetData.totalAudioFiles}</strong> | Genuine: <strong style="color:var(--status-low);">${datasetData.genuineCount}</strong> | Synthetic: <strong style="color:var(--status-critical);">${datasetData.syntheticCount}</strong> | Human Mimicry: <strong style="color:var(--status-medium);">${datasetData.humanMimicryCount}</strong> | Accents: <strong>${Object.keys(datasetData.accentBreakdown || {}).join(', ')}</strong>
      `;
    }

    tableBody.innerHTML = '';
    const sampleList = datasetData.sampleList || [];

    sampleList.forEach(sample => {
      const tr = document.createElement('tr');
      const isClone = sample.type === 'SYNTHETIC';
      const isMimicry = sample.type === 'HUMAN_MIMICRY';

      const badgeClass = isClone ? 'verdict-critical' : (isMimicry ? 'verdict-medium' : 'verdict-low');

      tr.innerHTML = `
        <td><strong>${sample.file}</strong><br><span style="font-size:0.75rem; color:var(--text-muted);">${sample.path}</span></td>
        <td><span class="verdict-badge ${badgeClass}" style="padding:0.2rem 0.6rem; font-size:0.72rem;">${sample.type}</span></td>
        <td>PyTorch 16kHz WAV Loader</td>
        <td>${sample.accent.toUpperCase()}</td>
        <td><button class="btn btn-secondary btn-test-sample" data-path="${sample.path}" style="padding:0.3rem 0.7rem; font-size:0.75rem;">▶ Analyze Real Sample File</button></td>
      `;

      tableBody.appendChild(tr);
    });

    document.querySelectorAll('.btn-test-sample').forEach(btn => {
      btn.addEventListener('click', async (e) => {
        const filePath = btn.getAttribute('data-path');
        btn.disabled = true;
        btn.innerText = "Fetching & Analyzing...";

        try {
          const fileRes = await fetch(filePath);
          if (fileRes.ok) {
            const blob = await fileRes.blob();
            const file = new File([blob], filePath.split('/').pop(), { type: 'audio/wav' });
            const analysisRes = await this.apiClient.analyzeAudioFile(file);
            btn.disabled = false;
            btn.innerText = "▶ Analyze Real Sample File";

            if (analysisRes.success && analysisRes.data) {
              this.renderAnalysisResult(analysisRes.data);
            }
          }
        } catch (err) {
          btn.disabled = false;
          btn.innerText = "▶ Analyze Real Sample File";
          alert("Error analyzing sample: " + err.message);
        }
      });
    });
  }

  async fetchLiveStats() {
    const res = await this.apiClient.getStats();
    if (res.success && res.data) {
      const stats = res.data;
      const elCalls = document.getElementById('stat-total-calls');
      const elBlocked = document.getElementById('stat-attacks-blocked');

      if (elCalls) elCalls.innerText = stats.totalCallsMonitored.toLocaleString();
      if (elBlocked) elBlocked.innerText = stats.attacksBlocked.toLocaleString();
    }
  }

  async loadEnrolledProfiles() {
    const res = await fetch('/api/v1/enroll');
    const json = await res.json();
    if (json.success && json.data) {
      const listContainer = document.getElementById('profiles-list-box');
      if (!listContainer) return;
      listContainer.innerHTML = '';
      json.data.forEach(p => {
        const div = document.createElement('div');
        div.className = 'caller-card';
        div.style.marginBottom = '0.75rem';
        div.innerHTML = `
          <div class="avatar-circle">${p.name.charAt(0)}</div>
          <div>
            <strong style="color:var(--text-main); font-size:0.9rem;">${p.name}</strong>
            <div style="font-size:0.78rem; color:var(--text-muted);">${p.organization}</div>
            <div style="font-size:0.72rem; color:var(--accent-cyan); margin-top:0.2rem;">Acoustic Vector Profile | Baseline F0: ${p.f0Mean || p.baselinePitchMean || 150} Hz</div>
          </div>
        `;
        listContainer.appendChild(div);
      });
    }
  }
}
