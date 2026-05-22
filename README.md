# DeepSeek OpenAI Adapter

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure API keys in `dbill_settings.json`

3. Run the server:
```bash
python app.py
```

## API Endpoints

- `POST /v1/chat/completions` - OpenAI-compatible chat endpoint
- `GET /v1/models` - List available models
- `GET /health` - Health check

## Configuration

Edit `dbill_settings.json` to set your OpenAI API key and DeepSeek runtime path.
