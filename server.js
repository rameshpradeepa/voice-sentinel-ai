const express = require('express');
const path = require('path');
const http = require('http');

const app = express();
let PORT = parseInt(process.env.PORT, 10) || 3000;
const FASTAPI_URL = process.env.FASTAPI_URL || 'http://127.0.0.1:8000';

app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));
app.use(express.static(path.join(__dirname, 'public')));

// Forward any API calls to authoritative FastAPI ML backend
app.use('/api', (req, res) => {
  const targetUrl = `${FASTAPI_URL}/api${req.url}`;
  const options = {
    method: req.method,
    headers: { ...req.headers, host: '127.0.0.1:8000' }
  };

  const proxyReq = http.request(targetUrl, options, (proxyRes) => {
    res.writeHead(proxyRes.statusCode, proxyRes.headers);
    proxyRes.pipe(res, { end: true });
  });

  proxyReq.on('error', (err) => {
    console.warn(`[!] Proxy notice: FastAPI server at ${FASTAPI_URL} not reachable.`, err.message);
    res.status(503).json({
      success: false,
      error: 'FASTAPI_SERVICE_UNAVAILABLE',
      message: 'FastAPI ML backend is starting or offline. Please start: uvicorn backend.main:app --port 8000'
    });
  });

  if (req.body && Object.keys(req.body).length > 0) {
    const bodyData = JSON.stringify(req.body);
    proxyReq.setHeader('Content-Type', 'application/json');
    proxyReq.setHeader('Content-Length', Buffer.byteLength(bodyData));
    proxyReq.write(bodyData);
  }

  proxyReq.end();
});

// Fallback to single page app index.html
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Function to start server with automatic port retry if EADDRINUSE occurs
function startServer(portToTry) {
  const server = app.listen(portToTry, () => {
    console.log(`=======================================================`);
    console.log(`🚀 VoiceSentinel AI Platform Running`);
    console.log(`📡 Frontend Static Server: http://localhost:${portToTry}`);
    console.log(`🤖 Authoritative FastAPI ML Proxy: ${FASTAPI_URL}`);
    console.log(`=======================================================`);
  });

  server.on('error', (err) => {
    if (err.code === 'EADDRINUSE') {
      console.warn(`[!] Port ${portToTry} is already in use. Retrying on port ${portToTry + 1}...`);
      startServer(portToTry + 1);
    } else {
      console.error('Server error:', err);
    }
  });
}

startServer(PORT);
