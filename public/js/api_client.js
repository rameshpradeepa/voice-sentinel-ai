/**
 * VoiceSentinel AI - FastAPI Backend REST API Client (v3.0)
 * Supports real audio file upload analysis, PyTorch model query, disk dataset exploration,
 * security action execution, and Blockchain audit logs.
 */

class ApiClient {
  constructor() {
    this.baseUrl = window.location.hostname ? `${window.location.protocol}//${window.location.hostname}:8000` : 'http://localhost:8000';
  }

  async getHealth() {
    try {
      const res = await fetch(`${this.baseUrl}/api/health`);
      return await res.json();
    } catch (err) {
      return { status: "offline", error: err.message };
    }
  }

  async analyzeStreamPayload(payload) {
    try {
      const res = await fetch(`${this.baseUrl}/api/v1/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      return await res.json();
    } catch (err) {
      return { success: false, error: err.message };
    }
  }

  async analyzeAudioFile(file, targetProfileId = null) {
    try {
      const formData = new FormData();
      formData.append('file', file);
      if (targetProfileId) formData.append('targetProfileId', targetProfileId);

      const res = await fetch(`${this.baseUrl}/api/v1/analyze-file`, {
        method: 'POST',
        body: formData
      });
      return await res.json();
    } catch (err) {
      return { success: false, error: err.message };
    }
  }

  async triggerSecurityAction(incidentId, action, riskScore, classification, audioHashSha256) {
    try {
      const res = await fetch(`${this.baseUrl}/api/v1/security/action`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ incidentId, action, riskScore, classification, audioHashSha256 })
      });
      return await res.json();
    } catch (err) {
      return { success: false, error: err.message };
    }
  }

  async getModelInfo() {
    try {
      const res = await fetch(`${this.baseUrl}/api/v1/model`);
      return await res.json();
    } catch (err) {
      return { success: false, error: err.message };
    }
  }

  async getDatasets() {
    try {
      const res = await fetch(`${this.baseUrl}/api/v1/datasets`);
      return await res.json();
    } catch (err) {
      return { success: false, error: err.message };
    }
  }

  async getStats() {
    try {
      const res = await fetch(`${this.baseUrl}/api/v1/stats`);
      return await res.json();
    } catch (err) {
      return { success: false, error: err.message };
    }
  }

  async testSipFilter(sipHeader, callerNumber) {
    try {
      const res = await fetch(`${this.baseUrl}/api/v1/telecom/sip-filter`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sipHeader, callerNumber, streamSampleRate: 16000 })
      });
      return await res.json();
    } catch (err) {
      return { success: false, error: err.message };
    }
  }

  async recordBlockchainAudit(incidentId, riskScore, classification, audioHashSha256) {
    try {
      const res = await fetch(`${this.baseUrl}/api/v1/audit`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ incidentId, riskScore, classification, audioHashSha256 })
      });
      return await res.json();
    } catch (err) {
      return { success: false, error: err.message };
    }
  }
}

window.ApiClient = ApiClient;
