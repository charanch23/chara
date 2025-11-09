/**
 * index.js (Hugging Face Inference)
 *
 * - Set environment variables:
 *   PROVIDER=huggingface
 *   HUGGINGFACE_API_TOKEN=hf_...
 *   PORT=4000
 *
 * - This backend exposes POST /api/generate { prompt, size, n }
 *   and returns: { images: [ dataURL-or-url, ... ] }
 *
 * Notes:
 * - Free-tier tokens exist, but there are usage limits and rate limits.
 * - If the HF model returns binary image bytes, we convert to base64 data URLs.
 * - Model chosen: stabilityai/stable-diffusion-2 (changeable).
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
const PROVIDER = (process.env.PROVIDER || 'huggingface').toLowerCase();
const HF_TOKEN = process.env.HUGGINGFACE_API_TOKEN;

// default model — change if you prefer another HF model (see note below)
const HF_MODEL = 'stabilityai/stable-diffusion-2';

if (PROVIDER !== 'huggingface') {
  console.warn('Configured provider is not huggingface. Set PROVIDER=huggingface in env.');
}
if (!HF_TOKEN) {
  console.warn('HUGGINGFACE_API_TOKEN not set in environment.');
}

// Helper to call Hugging Face Inference
async function callHuggingFaceModel(model, prompt, options = {}) {
  const url = `https://api-inference.huggingface.co/models/${encodeURIComponent(model)}`;

  // HF accepts JSON: { inputs: prompt, options: { wait_for_model: true }, parameters: {...} }
  // Some image models accept 'width'/'height' in parameters.
  const payload = {
    inputs: prompt,
    options: { wait_for_model: true },
    parameters: {}
  };

  // map size if provided (like "512x512")
  if (options.size) {
    const parts = String(options.size).split('x').map(s => parseInt(s.trim(), 10));
    if (parts.length === 2 && !isNaN(parts[0]) && !isNaN(parts[1])) {
      // Some HF models accept width/height in parameters
      payload.parameters.width = parts[0];
      payload.parameters.height = parts[1];
    }
  }
  // Add guidance/steps if you want (optional)
  if (options.steps) payload.parameters.num_inference_steps = options.steps;
  if (options.cfg_scale) payload.parameters.guidance_scale = options.cfg_scale;

  const resp = await axios.post(url, payload, {
    headers: {
      Authorization: `Bearer ${HF_TOKEN}`,
      Accept: 'application/json'
    },
    responseType: 'arraybuffer', // some models return raw bytes; handle both
    timeout: 120000
  });

  // Determine response type by content-type
  const contentType = resp.headers['content-type'] || '';

  // If HF returns JSON (some models), parse it
  if (contentType.includes('application/json')) {
    const text = Buffer.from(resp.data).toString('utf8');
    let parsed;
    try { parsed = JSON.parse(text); } catch (e) { parsed = text; }

    // Common shapes:
    // - Some spaces/models return { images: [ base64strings ] } or similar
    // - Some return { generated_image: 'data:...' } or { '0': 'data:...' }
    // Attempt to find base64 strings
    const images = [];

    function extractBase64(obj) {
      if (!obj) return;
      if (typeof obj === 'string') {
        // if looks like data:image or base64
        if (obj.startsWith('data:image')) images.push(obj);
        else if (/^[A-Za-z0-9+/]+={0,2}$/.test(obj) && obj.length > 100) {
          // looks like raw base64 without mime — assume png
          images.push('data:image/png;base64,' + obj);
        }
      } else if (Array.isArray(obj)) {
        obj.forEach(extractBase64);
      } else if (typeof obj === 'object') {
        Object.values(obj).forEach(extractBase64);
      }
    }

    extractBase64(parsed);

    // If nothing found, return the whole parsed object for debugging
    if (images.length === 0) {
      return { raw: parsed };
    }
    return { images };
  }

  // If binary image returned, convert to base64 data URL
  if (contentType.startsWith('image/')) {
    const base64 = Buffer.from(resp.data).toString('base64');
    const dataUrl = `data:${contentType};base64,${base64}`;
    return { images: [dataUrl] };
  }

  // Fallback: try to interpret resp as utf8 JSON
  try {
    const text = Buffer.from(resp.data).toString('utf8');
    const parsed = JSON.parse(text);
    // try to extract images
    if (parsed) {
      // find base64 or data urls
      const images = [];
      function extract(obj) {
        if (!obj) return;
        if (typeof obj === 'string' && obj.startsWith('data:image')) images.push(obj);
        else if (Array.isArray(obj)) obj.forEach(extract);
        else if (typeof obj === 'object') Object.values(obj).forEach(extract);
      }
      extract(parsed);
      if (images.length) return { images };
      return { raw: parsed };
    }
  } catch (e) {
    // last fallback
  }

  throw new Error('Unknown response format from Hugging Face inference');
}

// Main generate wrapper
async function generateWithHuggingFace({ prompt, size = '1024x1024', n = 1 }) {
  if (!HF_TOKEN) throw new Error('HUGGINGFACE_API_TOKEN not configured');

  const results = [];
  const count = Math.min(Math.max(parseInt(n || 1, 10), 1), 4); // 1..4 to be safe

  for (let i = 0; i < count; i++) {
    const out = await callHuggingFaceModel(HF_MODEL, prompt, { size });
    if (out.images) {
      results.push(...out.images);
    } else if (out.raw) {
      // Push JSON string for debugging (rare)
      results.push('data:application/json;base64,' + Buffer.from(JSON.stringify(out.raw)).toString('base64'));
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

    if (PROVIDER !== 'huggingface') {
      return res.status(400).json({ error: `Unsupported provider: ${PROVIDER}` });
    }

    const images = await generateWithHuggingFace({ prompt: prompt.toString(), size: size || '512x512', n: n || 1 });
    return res.json({ images });
  } catch (err) {
    let message = 'Generation failed';
    if (err.response && err.response.data) {
      // If HF returned an error body, attempt to parse and show message
      try {
        const text = Buffer.from(err.response.data).toString('utf8');
        const parsed = JSON.parse(text);
        message = parsed.error || JSON.stringify(parsed);
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

app.get('/', (req, res) => res.send('AI Image Backend (Hugging Face) — running'));

app.listen(PORT, () => {
  console.log(`Server listening on port ${PORT} (provider=${PROVIDER})`);
});
