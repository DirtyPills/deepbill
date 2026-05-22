#!/usr/bin/env python3
"""
OpenAI API Adapter Module
"""

import json
import logging
from typing import Dict, Any, List, Optional, Iterator
import requests

logger = logging.getLogger(__name__)

class OpenAIAdapter:
    """Adapter for OpenAI API compatibility"""
    
    def __init__(self, config: Dict[str, Any]):
        self.api_key = config.get('api_key', '')
        self.base_url = config.get('base_url', 'https://api.openai.com/v1')
        self.timeout = config.get('timeout', 60)
        
    def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = 'auto'
    ) -> Dict[str, Any]:
        """Send chat completion request to OpenAI"""
        url = f"{self.base_url}/chat/completions"
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        payload = {
            'model': model,
            'messages': messages,
            'temperature': temperature,
            'max_tokens': max_tokens,
            'stream': False
        }
        
        if tools:
            payload['tools'] = tools
            payload['tool_choice'] = tool_choice
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    def stream_chat_completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = 'auto'
    ) -> Iterator[Dict[str, Any]]:
        """Stream chat completion from OpenAI"""
        url = f"{self.base_url}/chat/completions"
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        payload = {
            'model': model,
            'messages': messages,
            'temperature': temperature,
            'max_tokens': max_tokens,
            'stream': True
        }
        
        if tools:
            payload['tools'] = tools
            payload['tool_choice'] = tool_choice
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=self.timeout, stream=True)
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data = line[6:]
                        if data != '[DONE]':
                            yield json.loads(data)
        except requests.RequestException as e:
            logger.error(f"OpenAI streaming error: {e}")
            raise
