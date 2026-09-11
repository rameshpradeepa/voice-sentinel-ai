/**
 * VoiceSentinel AI - Spectrogram & Oscilloscope Canvas Renderer
 * Renders real-time 2D Spectrogram waterfall and Oscilloscope waveform graphics.
 */

class SpectrogramRenderer {
  constructor(spectrogramCanvasId, oscilloscopeCanvasId) {
    this.specCanvas = document.getElementById(spectrogramCanvasId);
    this.oscCanvas = document.getElementById(oscilloscopeCanvasId);

    this.specCtx = this.specCanvas ? this.specCanvas.getContext('2d') : null;
    this.oscCtx = this.oscCanvas ? this.oscCanvas.getContext('2d') : null;

    // Color gradient palette for spectral intensity (0 to 255)
    this.colorPalette = this.createSpectralPalette();

    // Off-screen canvas for smooth waterfall scrolling
    this.tempCanvas = document.createElement('canvas');
    this.tempCtx = this.tempCanvas.getContext('2d');
  }

  createSpectralPalette() {
    const palette = [];
    for (let i = 0; i < 256; i++) {
      let r = 0, g = 0, b = 0;
      if (i < 32) {
        // Deep Space Navy -> Dark Blue
        r = 7; g = 9; b = Math.floor(i * 4 + 19);
      } else if (i < 96) {
        // Cyan / Blue transition
        const t = (i - 32) / 64;
        r = 0; g = Math.floor(t * 240); b = 255;
      } else if (i < 192) {
        // Cyan -> Neon Purple
        const t = (i - 96) / 96;
        r = Math.floor(t * 138); g = Math.floor((1 - t) * 240 + 43); b = 226;
      } else {
        // Purple -> Glowing Red / Coral (High Intensity Phase Artifacts)
        const t = (i - 192) / 64;
        r = 255; g = Math.floor((1 - t) * 100); b = Math.floor((1 - t) * 100);
      }
      palette.push(`rgb(${r},${g},${b})`);
    }
    return palette;
  }

  renderFrame(freqData, timeData) {
    if (this.specCtx && freqData) {
      this.drawWaterfall(freqData);
    }
    if (this.oscCtx && timeData) {
      this.drawOscilloscope(timeData);
    }
  }

  drawWaterfall(freqData) {
    const width = this.specCanvas.width;
    const height = this.specCanvas.height;

    // Resize temp canvas if needed
    if (this.tempCanvas.width !== width || this.tempCanvas.height !== height) {
      this.tempCanvas.width = width;
      this.tempCanvas.height = height;
    }

    // 1. Copy current canvas to temp canvas
    this.tempCtx.drawImage(this.specCanvas, 0, 0);

    // 2. Shift canvas left by 2 pixels for scrolling waterfall effect
    this.specCtx.drawImage(this.tempCanvas, -2, 0);

    // 3. Draw new slice at the right edge
    const sliceWidth = 2;
    const x = width - sliceWidth;
    const binCount = freqData.length;
    const barHeight = height / binCount;

    for (let i = 0; i < binCount; i++) {
      const val = freqData[i];
      const color = this.colorPalette[val] || this.colorPalette[0];
      this.specCtx.fillStyle = color;
      
      // Invert Y so low frequencies are at bottom, high frequencies at top
      const y = height - (i + 1) * barHeight;
      this.specCtx.fillRect(x, y, sliceWidth, barHeight + 0.5);
    }
  }

  drawOscilloscope(timeData) {
    const width = this.oscCanvas.width;
    const height = this.oscCanvas.height;

    this.oscCtx.fillStyle = '#04060E';
    this.oscCtx.fillRect(0, 0, width, height);

    // Grid lines
    this.oscCtx.strokeStyle = 'rgba(45, 64, 105, 0.2)';
    this.oscCtx.lineWidth = 1;
    this.oscCtx.beginPath();
    this.oscCtx.moveTo(0, height / 2);
    this.oscCtx.lineTo(width, height / 2);
    this.oscCtx.stroke();

    // Waveform line
    this.oscCtx.lineWidth = 2;
    this.oscCtx.strokeStyle = '#00F0FF';
    this.oscCtx.shadowBlur = 8;
    this.oscCtx.shadowColor = '#00F0FF';
    this.oscCtx.beginPath();

    const sliceWidth = width / timeData.length;
    let x = 0;

    for (let i = 0; i < timeData.length; i++) {
      const v = timeData[i] / 128.0;
      const y = (v * height) / 2;

      if (i === 0) {
        this.oscCtx.moveTo(x, y);
      } else {
        this.oscCtx.lineTo(x, y);
      }

      x += sliceWidth;
    }

    this.oscCtx.lineTo(width, height / 2);
    this.oscCtx.stroke();
    this.oscCtx.shadowBlur = 0;
  }
}

window.SpectrogramRenderer = SpectrogramRenderer;
