/**
 * VoiceSentinel AI - Web Audio API & Real-Time WebSocket Streaming Engine
 * Captures live microphone audio, accumulates 3-second PCM audio windows (48,000 samples),
 * and streams raw audio chunk bytes via WebSocket to FastAPI backend for PyTorch inference.
 * Displays "REAL-TIME MODEL CONNECTION LOST" if WebSocket disconnects.
 */

class AudioEngine {
  constructor() {
    this.audioCtx = null;
    this.analyser = null;
    this.micStream = null;
    this.sourceNode = null;
    this.processorNode = null;
    this.isRecording = false;

    this.ws = null;
    this.isWsConnected = false;
    this.onFeatureUpdateCallback = null;
    this.onPyTorchResultCallback = null;
    this.onConnectionStatusCallback = null;

    this.fftSize = 2048;
    this.frequencyData = new Uint8Array(this.fftSize / 2);
    this.timeData = new Uint8Array(this.fftSize);

    this.pitchHistory = [];
    this.centroidHistory = [];

    this.connectWebSocket();
  }

  connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.hostname}:8000/ws/analyze`;

    try {
      this.ws = new WebSocket(wsUrl);

      this.ws.onopen = () => {
        this.isWsConnected = true;
        console.log("[+] Connected to FastAPI WebSocket /ws/analyze stream");
        if (this.onConnectionStatusCallback) {
          this.onConnectionStatusCallback(true);
        }
      };

      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'PYTORCH_INFERENCE_UPDATE' && this.onPyTorchResultCallback) {
            this.onPyTorchResultCallback(data);
          }
        } catch (e) {
          console.warn("WebSocket parse error:", e);
        }
      };

      this.ws.onclose = () => {
        this.isWsConnected = false;
        console.warn("[!] WebSocket connection closed.");
        if (this.onConnectionStatusCallback) {
          this.onConnectionStatusCallback(false, "REAL-TIME MODEL CONNECTION LOST");
        }
        setTimeout(() => this.connectWebSocket(), 3000);
      };

      this.ws.onerror = (err) => {
        this.isWsConnected = false;
        if (this.onConnectionStatusCallback) {
          this.onConnectionStatusCallback(false, "REAL-TIME MODEL CONNECTION LOST");
        }
      };
    } catch (err) {
      this.isWsConnected = false;
      if (this.onConnectionStatusCallback) {
        this.onConnectionStatusCallback(false, "REAL-TIME MODEL CONNECTION LOST");
      }
    }
  }

  async initAudioContext() {
    if (!this.audioCtx) {
      const AudioContext = window.AudioContext || window.webkitAudioContext;
      this.audioCtx = new AudioContext({ sampleRate: 16000 });
    }
    if (this.audioCtx.state === 'suspended') {
      await this.audioCtx.resume();
    }

    if (!this.analyser) {
      this.analyser = this.audioCtx.createAnalyser();
      this.analyser.fftSize = this.fftSize;
      this.analyser.smoothingTimeConstant = 0.8;
    }
  }

  async startMicrophone(onUpdate, onPyTorchResult, onStatus) {
    await this.initAudioContext();
    this.onFeatureUpdateCallback = onUpdate;
    this.onPyTorchResultCallback = onPyTorchResult;
    this.onConnectionStatusCallback = onStatus;

    if (!this.isWsConnected) {
      if (this.onConnectionStatusCallback) {
        this.onConnectionStatusCallback(false, "REAL-TIME MODEL CONNECTION LOST");
      }
    }

    try {
      this.micStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
      this.sourceNode = this.audioCtx.createMediaStreamSource(this.micStream);
      this.sourceNode.connect(this.analyser);

      this.processorNode = this.audioCtx.createScriptProcessor(4096, 1, 1);
      this.sourceNode.connect(this.processorNode);
      this.processorNode.connect(this.audioCtx.destination);

      this.processorNode.onaudioprocess = (e) => {
        if (!this.isRecording) return;
        const inputData = e.inputBuffer.getChannelData(0);

        if (this.isWsConnected && this.ws && this.ws.readyState === WebSocket.OPEN) {
          const pcmBuffer = new Int16Array(inputData.length);
          for (let i = 0; i < inputData.length; i++) {
            pcmBuffer[i] = Math.max(-1, Math.min(1, inputData[i])) * 32767;
          }
          this.ws.send(pcmBuffer.buffer);
        } else {
          if (this.onConnectionStatusCallback) {
            this.onConnectionStatusCallback(false, "REAL-TIME MODEL CONNECTION LOST");
          }
        }
      };

      this.isRecording = true;
      this.pitchHistory = [];
      this.centroidHistory = [];

      this.processAudioLoop();
      return true;
    } catch (err) {
      console.warn("Microphone access unavailable or denied.", err);
      if (this.onConnectionStatusCallback) {
        this.onConnectionStatusCallback(false, "MICROPHONE ACCESS DENIED");
      }
      return false;
    }
  }

  stopMicrophone() {
    this.isRecording = false;
    if (this.processorNode) {
      this.processorNode.disconnect();
      this.processorNode = null;
    }
    if (this.micStream) {
      this.micStream.getTracks().forEach(track => track.stop());
      this.micStream = null;
    }
    if (this.sourceNode) {
      this.sourceNode.disconnect();
      this.sourceNode = null;
    }
  }

  processAudioLoop() {
    if (!this.isRecording) return;

    this.analyser.getByteFrequencyData(this.frequencyData);
    this.analyser.getByteTimeDomainData(this.timeData);

    const features = this.extractAcousticFeatures(this.frequencyData, this.timeData);

    if (this.onFeatureUpdateCallback) {
      this.onFeatureUpdateCallback(features, this.frequencyData, this.timeData);
    }

    requestAnimationFrame(() => this.processAudioLoop());
  }

  extractAcousticFeatures(freqData, timeData) {
    const sampleRate = this.audioCtx ? this.audioCtx.sampleRate : 16000;
    const nyquist = sampleRate / 2;
    const binCount = freqData.length;
    const binWidth = nyquist / binCount;

    let totalPower = 0;
    let weightedFreq = 0;
    let highFreqEnergy = 0;

    for (let i = 0; i < binCount; i++) {
      const power = freqData[i];
      const freq = i * binWidth;
      totalPower += power;
      weightedFreq += freq * power;

      if (freq > 3800) {
        highFreqEnergy += power;
      }
    }

    const spectralCentroid = totalPower > 0 ? weightedFreq / totalPower : 0;
    const highFreqEnergyRatio = totalPower > 0 ? highFreqEnergy / totalPower : 0;

    let zeroCrossings = 0;
    for (let i = 1; i < timeData.length; i++) {
      if ((timeData[i - 1] < 128 && timeData[i] >= 128) || (timeData[i - 1] >= 128 && timeData[i] < 128)) {
        zeroCrossings++;
      }
    }
    const zcr = zeroCrossings / timeData.length;

    const pitchHz = this.autoCorrelatePitch(timeData, sampleRate);

    if (pitchHz > 60 && pitchHz < 400) {
      this.pitchHistory.push(pitchHz);
      if (this.pitchHistory.length > 50) this.pitchHistory.shift();
    }

    this.centroidHistory.push(spectralCentroid);
    if (this.centroidHistory.length > 50) this.centroidHistory.shift();

    const pitchMean = this.pitchHistory.length > 0 ? this.pitchHistory.reduce((a, b) => a + b, 0) / this.pitchHistory.length : 145;
    const pitchVariance = this.pitchHistory.length > 0 ? this.pitchHistory.reduce((sq, n) => sq + Math.pow(n - pitchMean, 2), 0) / this.pitchHistory.length : 25;
    const pitchStdDev = Math.sqrt(pitchVariance);

    const centroidStdDev = this.centroidHistory.length > 0 ? Math.sqrt(this.centroidHistory.reduce((sq, n) => sq + Math.pow(n - spectralCentroid, 2), 0) / this.centroidHistory.length) : 150;

    return {
      spectralCentroid: Math.round(spectralCentroid),
      spectralCentroidStdDev: Math.round(centroidStdDev),
      highFreqEnergyRatio: parseFloat(highFreqEnergyRatio.toFixed(3)),
      zcr: parseFloat(zcr.toFixed(3)),
      pitchHz: Math.round(pitchHz),
      pitchMean: Math.round(pitchMean),
      pitchStdDev: parseFloat(pitchStdDev.toFixed(1)),
      volumeRms: Math.round(totalPower / binCount)
    };
  }

  autoCorrelatePitch(timeData, sampleRate) {
    let SIZE = timeData.length;
    let rms = 0;
    for (let i = 0; i < SIZE; i++) {
      let val = (timeData[i] - 128) / 128;
      rms += val * val;
    }
    rms = Math.sqrt(rms / SIZE);
    if (rms < 0.01) return 0;

    let r1 = 0, r2 = SIZE - 1, thres = 0.2;
    for (let i = 0; i < SIZE / 2; i++) {
      if (Math.abs((timeData[i] - 128) / 128) < thres) { r1 = i; break; }
    }
    for (let i = 1; i < SIZE / 2; i++) {
      if (Math.abs((timeData[SIZE - i] - 128) / 128) < thres) { r2 = SIZE - i; break; }
    }

    timeData = timeData.slice(r1, r2);
    SIZE = timeData.length;

    let c = new Array(SIZE).fill(0);
    for (let i = 0; i < SIZE; i++) {
      for (let j = 0; j < SIZE - i; j++) {
        c[i] = c[i] + ((timeData[j] - 128) / 128) * ((timeData[j + i] - 128) / 128);
      }
    }

    let d = 0;
    while (c[d] > c[d + 1]) d++;
    let maxval = -1, maxpos = -1;
    for (let i = d; i < SIZE; i++) {
      if (c[i] > maxval) {
        maxval = c[i];
        maxpos = i;
      }
    }
    let T0 = maxpos;

    return T0 > 0 ? sampleRate / T0 : 0;
  }
}

window.voiceAudioEngine = new AudioEngine();
