/**
 * VoiceSentinel AI - FastAPI Backend REST API Client (v3.1)
 * Explicitly connects to http://127.0.0.1:8000 for backend endpoints.
 */

class ApiClient {
  constructor() {
    this.baseUrl = 'http://127.0.0.1:8000';
  }

  async getHealth() {
    try {
      const res = await fetch(`${this.baseUrl}/api/health`);
      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        return { status: "offline", error: errJson.detail || `HTTP ${res.status}` };
      }
      return await res.json();
    } catch (err) {
      return {
        status: "offline",
        error: "BACKEND_CONNECTION_FAILED",
        message: "Unable to connect to FastAPI backend at http://127.0.0.1:8000"
      };
    }
  }

  async analyzeStreamPayload(payload) {
    try {
      const res = await fetch(`${this.baseUrl}/api/v1/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        return {
          success: false,
          error: errJson.error || "HTTP_ERROR",
          message: errJson.message || errJson.detail || `HTTP Error ${res.status}`
        };
      }
      return await res.json();
    } catch (err) {
      return {
        success: false,
        error: "BACKEND_CONNECTION_FAILED",
        message: "Unable to connect to FastAPI backend at http://127.0.0.1:8000"
      };
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

      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        return {
          success: false,
          error: errJson.error || "HTTP_ERROR",
          message: errJson.message || errJson.detail || `HTTP Error ${res.status}: ${res.statusText}`,
          status: res.status
        };
      }

      return await res.json();
    } catch (err) {
      return {
        success: false,
        error: "BACKEND_CONNECTION_FAILED",
        message: "Unable to connect to FastAPI backend at http://127.0.0.1:8000"
      };
    }
  }

  async triggerSecurityAction(incidentId, action, riskScore, classification, audioHashSha256) {
    try {
      const res = await fetch(`${this.baseUrl}/api/v1/security/action`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ incidentId, action, riskScore, classification, audioHashSha256 })
      });
      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        return { success: false, error: errJson.error || "HTTP_ERROR", message: errJson.message || errJson.detail };
      }
      return await res.json();
    } catch (err) {
      return {
        success: false,
        error: "BACKEND_CONNECTION_FAILED",
        message: "Unable to connect to FastAPI backend at http://127.0.0.1:8000"
      };
    }
  }

  async getModelInfo() {
    try {
      const res = await fetch(`${this.baseUrl}/api/v1/model`);
      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        return { success: false, error: errJson.error || "HTTP_ERROR", message: errJson.message || errJson.detail };
      }
      return await res.json();
    } catch (err) {
      return {
        success: false,
        error: "BACKEND_CONNECTION_FAILED",
        message: "Unable to connect to FastAPI backend at http://127.0.0.1:8000"
      };
    }
  }

  async getDatasets() {
    try {
      const res = await fetch(`${this.baseUrl}/api/v1/datasets`);
      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        return { success: false, error: errJson.error || "HTTP_ERROR", message: errJson.message || errJson.detail };
      }
      return await res.json();
    } catch (err) {
      return {
        success: false,
        error: "BACKEND_CONNECTION_FAILED",
        message: "Unable to connect to FastAPI backend at http://127.0.0.1:8000"
      };
    }
  }

  async getStats() {
    try {
      const res = await fetch(`${this.baseUrl}/api/v1/stats`);
      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        return { success: false, error: errJson.error || "HTTP_ERROR", message: errJson.message || errJson.detail };
      }
      return await res.json();
    } catch (err) {
      return {
        success: false,
        error: "BACKEND_CONNECTION_FAILED",
        message: "Unable to connect to FastAPI backend at http://127.0.0.1:8000"
      };
    }
  }

  async testSipFilter(sipHeader, callerNumber) {
    try {
      const res = await fetch(`${this.baseUrl}/api/v1/telecom/sip-filter`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sipHeader, callerNumber, streamSampleRate: 16000 })
      });
      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        return { success: false, error: errJson.error || "HTTP_ERROR", message: errJson.message || errJson.detail };
      }
      return await res.json();
    } catch (err) {
      return {
        success: false,
        error: "BACKEND_CONNECTION_FAILED",
        message: "Unable to connect to FastAPI backend at http://127.0.0.1:8000"
      };
    }
  }

  async recordBlockchainAudit(incidentId, riskScore, classification, audioHashSha256) {
    try {
      const res = await fetch(`${this.baseUrl}/api/v1/audit`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ incidentId, riskScore, classification, audioHashSha256 })
      });
      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        return { success: false, error: errJson.error || "HTTP_ERROR", message: errJson.message || errJson.detail };
      }
      return await res.json();
    } catch (err) {
      return {
        success: false,
        error: "BACKEND_CONNECTION_FAILED",
        message: "Unable to connect to FastAPI backend at http://127.0.0.1:8000"
      };
    }
  }
}

window.ApiClient = ApiClient;
