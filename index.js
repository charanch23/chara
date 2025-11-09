/**
 * index.js (Replicate version)
 *
 * Usage:
 * - Set environment variables:
 *   PROVIDER=replicate
 *   REPLICATE_API_TOKEN=<your_replicate_token>
 *   PORT=4000
 *
 * - Deploy to Render (or run locally). Your frontend does not need any change.
 *
 * Notes:
 * - This uses Replicate's /v1/models/{owner}/{name}/versions to pick a version,
 *   then posts a prediction and polls until completion.
 * - Some Replicate models expect different input keys; the chosen stability model
 *   accepts a `prompt`. Adjust the modelName if you want a different one.
 */

require('dotenv').config();
const express = require('express');
const axios = require('axios');
const cors = require('cors');
const rateLimit = require('express-rate-limit');
const helmet = require('helmet');

const app = express();
app.use(helmet());
app.use(express.json({ limit: '10mb' }));

const FRONTEND_ORIGIN = process.env.FRONTEND_ORIGIN || '*';
app.use(cors({ origin: FRONTEND_ORIGIN }));

const limiter = rateLimit({
  windowMs: 60 * 1000,
  max: 40,
  message: { error: 'Too many requests, please slow down.' }
});
app.use(limiter);

const PORT = process.env.PORT || 4000;
const PROVIDER = (process.env.PROVIDER || 'replicate').toLowerCase();

if (PROVIDER !== 'replicate') {
  console.warn('Warning: backend configured for replicate. Set PROVIDER=replicate in env.');
}

if (!process.env.REPLICATE_API_TOKEN) {
  console.warn('Warning: REPLICATE_API_TOKEN is not set. Set it in Render env or local .env for testing.');
}

const REPLICATE_API_BASE = 'https://api.replicate.com/v1';
const REPLICATE_TOKEN = process.env.REPLICATE_API_TOKEN;

// Choose model (owner/name). You can change this to another replicate model later.
const MODEL_OWNER = 'stability-ai';
const MODEL_NAME = 'stable-diffusion';

// Helper: get latest version id of a model
async function getLatestModelVersion(owner = MODEL_OWNER, name = MODEL_NAME) {
  const url = `${REPLICATE_API_BASE}/models/${encodeURIComponent(owner)}/${encodeURIComponent(name)}/versions`;
  const resp = await axios.get(url, {
    headers: { Authorization: `Token ${REPLICATE_TOKEN}` },
    timeout: 120000
  });
  const versions = resp.data?.results || resp.data || [];
  if (!versions || versions.length === 0) throw new Error('No Replicate model versions found');
  // Prefer first version (API returns newest first commonly)
  return versions[0].id || versions[0].version || versions[0].uid || versions[0].name;
}

// Create a prediction (single run)
async function createPrediction(versionId, input) {
  const url = `${REPLICATE_API_BASE}/predictions`;
  const resp = await axios.post(
    url,
    { version: versionId, input },
    { headers: { Authorization: `Token ${REPLICATE_TOKEN}`, 'Content-Type': 'application/json' }, timeout: 120000 }
  );
  return resp.data;
}

// Poll prediction until done or failed
async function pollPrediction(predictionId, intervalMs = 1500, maxAttempts = 80) {
  const url = `${REPLICATE_API_BASE}/predictions/${predictionId}`;
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    const r = await axios.get(url, {
      headers: { Authorization: `Token ${REPLICATE_TOKEN}` },
      timeout: 120000
    });
    const p = r.data;
    if (p.status === 'succeeded') return p;
    if (p.status === 'failed') throw new Error(`Replicate prediction failed: ${JSON.stringify(p.error || p)}`);
    // otherwise: 'starting', 'processing'
    await new Promise(res => setTimeout(res, intervalMs));
  }
  throw new Error('Replicate prediction timed out');
}

/**
 * Generate images using Replicate.
 * Returns an array of image URLs (or data URIs if model returns base64).
 *
 * Note: Many Replicate models accept input like { prompt }.
 * Some models accept width/height/num_inference_steps/guidance_scale. We pass only prompt and optional size.
 */
async function generateWithReplicate({ prompt, size = '1024x1024', n = 1 }) {
  if (!REPLICATE_TOKEN) throw new Error('REPLICATE_API_TOKEN not configured');

  // Map size string like "1024x1024" to width/height numbers if model supports it
  let width = undefined, height = undefined;
  if (typeof size === 'string' && size.includes('x')) {
    const parts = size.split('x').map(s => parseInt(s.trim(), 10));
    if (parts.length === 2 && !isNaN(parts[0]) && !isNaN(parts[1])) {
      width = parts[0];
      height = parts[1];
    }
  }

  // Get model version id once
  const versionId = await getLatestModelVersion(MODEL_OWNER, MODEL_NAME);

  // Replicate predictions are usually created one per call. If n > 1 we'll create sequential predictions.
  const results = [];
  const limitCount = Math.min(Math.max(parseInt(n || 1, 10), 1), 6); // avoid excessive parallel calls

  for (let i = 0; i < limitCount; i++) {
    // Input object depends on model. For Stability's sd model, 'prompt' is accepted.
    const input = { prompt };

    // Some versions accept width/height; include if parsed successfully
    if (width && height) {
      input.width = width;
      input.height = height;
    }

    // Create prediction
    const prediction = await createPrediction(versionId, input);

    // Poll until finished
    const finished = await pollPrediction(prediction.id);

    // finished.output might be an array of URLs (common)
    // some models return base64; if so, we handle that too.
    const out = finished.output;
    if (!out) {
      // no output, push the whole finished object for debugging
      results.push(JSON.stringify(finished));
    } else if (Array.isArray(out)) {
      // push each URL/data
      out.forEach(o => results.push(o));
    } else {
      results.push(out);
    }
  }

  return results;
}

// API endpoint
app.post('/api/generate', async (req, res) => {
  try {
    const { prompt, size, n } = req.body || {};
    if (!prompt || !prompt.toString().trim()) {
      return res.status(400).json({ error: 'Missing prompt in request body' });
    }

    const count = Math.min(Math.max(parseInt(n || 1, 10), 1), 6); // 1..6
    const imgSize = size || '1024x1024';

    if (PROVIDER !== 'replicate') {
      return res.status(400).json({ error: `Unsupported provider: ${PROVIDER}` });
    }

    const images = await generateWithReplicate({ prompt: prompt.toString(), size: imgSize, n: count });
    return res.json({ images });
  } catch (err) {
    // try to extract useful message without leaking secrets
    let message = 'Generation failed';
    if (err.response && err.response.data) {
      try {
        message = err.response.data?.error || JSON.stringify(err.response.data);
      } catch (e) {
        message = String(err.response.data);
      }
    } else if (err.message) {
      message = err.message;
    }
    console.error('Generate error:', message);
    return res.status(500).json({ error: message });
  }
});

app.get('/', (req, res) => {
  res.send('AI Image Backend (Replicate) — running');
});

app.listen(PORT, () => {
  console.log(`Server listening on port ${PORT} (provider=${PROVIDER})`);
});
