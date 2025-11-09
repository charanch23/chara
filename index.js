/**
 * index.js
 * Simple Express backend that proxies requests to OpenAI Images API.
 *
 * IMPORTANT:
 * - Do NOT commit real API keys. Use Render environment variables.
 * - If deploying to Render, add OPENAI_API_KEY and PROVIDER=openai in Render's Environment.
 */

require('dotenv').config();
const express = require('express');
const axios = require('axios');
const cors = require('cors');
const rateLimit = require('express-rate-limit');
const helmet = require('helmet');

const app = express();

// SECURITY / PARSING
app.use(helmet());
app.use(express.json({ limit: '5mb' }));

// CORS: in production, replace '*' with your frontend domain (e.g. 'https://example.com')
const FRONTEND_ORIGIN = process.env.FRONTEND_ORIGIN || '*';
app.use(cors({ origin: FRONTEND_ORIGIN }));

// BASIC RATE LIMIT (tweak for your needs)
const limiter = rateLimit({
  windowMs: 60 * 1000, // 1 minute
  max: 30, // max requests per IP per window
  message: { error: 'Too many requests, please slow down.' }
});
app.use(limiter);

const PORT = process.env.PORT || 4000;
const PROVIDER = (process.env.PROVIDER || 'openai').toLowerCase();

// warn if key missing (but don't log the key)
if (!process.env.OPENAI_API_KEY) {
  console.warn('Warning: OPENAI_API_KEY is not set. Set it in Render env or local .env for testing.');
}

// Adapter for OpenAI Images
async function generateWithOpenAI({ prompt, size = '1024x1024', n = 1 }) {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) throw new Error('OPENAI_API_KEY not configured');

  // OpenAI Images endpoint (v1/images/generations)
  const url = 'https://api.openai.com/v1/images/generations';

  const payload = {
    prompt,
    n: Number(n) || 1,
    size
  };

  const resp = await axios.post(url, payload, {
    headers: {
      Authorization: `Bearer ${apiKey}`,
      'Content-Type': 'application/json'
    },
    timeout: 120000
  });

  const items = resp.data?.data || [];
  const results = items.map(item => {
    if (item.b64_json) {
      return `data:image/png;base64,${item.b64_json}`;
    } else if (item.url) {
      return item.url;
    } else {
      return null;
    }
  }).filter(Boolean);

  return results;
}

// API endpoint
app.post('/api/generate', async (req, res) => {
  try {
    const { prompt, size, n } = req.body || {};
    if (!prompt || !prompt.toString().trim()) {
      return res.status(400).json({ error: 'Missing prompt in request body' });
    }

    // Safety caps
    const count = Math.min(Math.max(parseInt(n || 1, 10), 1), 10); // between 1 and 10
    const imgSize = size || '1024x1024';

    if (PROVIDER !== 'openai') {
      return res.status(400).json({ error: `Unsupported provider: ${PROVIDER}` });
    }

    const images = await generateWithOpenAI({ prompt: prompt.toString(), size: imgSize, n: count });
    return res.json({ images });
  } catch (err) {
    // Try to extract helpful message
    let message = 'Generation failed';
    if (err.response && err.response.data) {
      // don't leak secrets — show the provider response
      message = err.response.data?.error?.message || JSON.stringify(err.response.data);
    } else if (err.message) {
      message = err.message;
    }
    console.error('Generate error:', message);
    return res.status(500).json({ error: message });
  }
});

app.get('/', (req, res) => {
  res.send('AI Image Backend — running');
});

app.listen(PORT, () => {
  console.log(`Server listening on port ${PORT} (provider=${PROVIDER})`);
});
