#!/usr/bin/env python3
"""
DeepSeek OpenAI-Compatible Adapter Server
"""

import json
import logging
import os
import sys
from typing import Optional, Dict, Any, List, Union
from datetime import datetime

from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import requests

# Import local modules
from openai_adapter import OpenAIAdapter
from deepseek_runtime import DeepSeekRuntime
from tool_call_parser import parse_assistant_message, extract_tool_calls

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load configuration
CONFIG_FILE = 'dbill_settings.json'

def load_config():
    """Load configuration from JSON file"""
    try:
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"Config file {CONFIG_FILE} not found, using defaults")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing config file: {e}")
        return {}

config = load_config()

# Initialize adapters
openai_adapter = OpenAIAdapter(config.get('openai', {}))
deepseek_runtime = DeepSeekRuntime(config.get('deepseek', {}))

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """OpenAI-compatible chat completions endpoint"""
    try:
        data = request.get_json()
        logger.info(f"Received chat completion request: {data.get('model', 'unknown')}")
        
        # Check if streaming is requested
        stream = data.get('stream', False)
        
        if stream:
            return Response(
                stream_with_context(handle_streaming_response(data)),
                content_type='text/event-stream',
                headers={'Cache-Control': 'no-cache'}
            )
        else:
            response = handle_non_streaming_response(data)
            return jsonify(response)
    except Exception as e:
        logger.error(f"Error in chat_completions: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

def handle_non_streaming_response(data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle non-streaming chat completion request"""
    model = data.get('model', 'deepseek-chat')
    messages = data.get('messages', [])
    temperature = data.get('temperature', 0.7)
    max_tokens = data.get('max_tokens', 4096)
    tools = data.get('tools')
    tool_choice = data.get('tool_choice', 'auto')
    
    # Route to appropriate backend
    if 'deepseek' in model.lower():
        response = deepseek_runtime.chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice
        )
    else:
        response = openai_adapter.chat_completion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice
        )
    
    return response

def handle_streaming_response(data: Dict[str, Any]):
    """Generate streaming response chunks"""
    model = data.get('model', 'deepseek-chat')
    messages = data.get('messages', [])
    temperature = data.get('temperature', 0.7)
    max_tokens = data.get('max_tokens', 4096)
    tools = data.get('tools')
    tool_choice = data.get('tool_choice', 'auto')
    
    if 'deepseek' in model.lower():
        for chunk in deepseek_runtime.stream_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice
        ):
            yield f"data: {json.dumps(chunk)}\n\n"
    else:
        for chunk in openai_adapter.stream_chat_completion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice
        ):
            yield f"data: {json.dumps(chunk)}\n\n"
    
    yield "data: [DONE]\n\n"

@app.route('/v1/models', methods=['GET'])
def list_models():
    """List available models"""
    models = [
        {
            "id": "deepseek-chat",
            "object": "model",
            "created": int(datetime.now().timestamp()),
            "owned_by": "deepseek"
        },
        {
            "id": "gpt-4",
            "object": "model",
            "created": int(datetime.now().timestamp()),
            "owned_by": "openai"
        },
        {
            "id": "gpt-3.5-turbo",
            "object": "model",
            "created": int(datetime.now().timestamp()),
            "owned_by": "openai"
        }
    ]
    return jsonify({"object": "list", "data": models})

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'false').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)
