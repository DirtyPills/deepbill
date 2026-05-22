#!/usr/bin/env python3
"""
DeepSeek Runtime Integration Module
"""

import json
import logging
import subprocess
import tempfile
from typing import Dict, Any, List, Optional, Iterator
import os

logger = logging.getLogger(__name__)

class DeepSeekRuntime:
    """Interface with DeepSeek runtime for local execution"""
    
    def __init__(self, config: Dict[str, Any]):
        self.deepseek_path = config.get('deepseek_path', 'deepseek')
        self.working_dir = config.get('working_dir', '.')
        
    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = 'auto'
    ) -> Dict[str, Any]:
        """Execute chat completion using DeepSeek runtime"""
        # Create temporary input file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            input_data = {
                'messages': messages,
                'temperature': temperature,
                'max_tokens': max_tokens
            }
            if tools:
                input_data['tools'] = tools
                input_data['tool_choice'] = tool_choice
            
            json.dump(input_data, f)
            input_file = f.name
        
        try:
            # Execute deepseek runtime
            cmd = [self.deepseek_path, 'chat', '--input', input_file]
            result = subprocess.run(
                cmd,
                cwd=self.working_dir,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                logger.error(f"DeepSeek runtime error: {result.stderr}")
                raise RuntimeError(f"DeepSeek execution failed: {result.stderr}")
            
            # Parse output
            output = json.loads(result.stdout)
            return self._format_response(output)
            
        finally:
            # Cleanup
            os.unlink(input_file)
    
    def stream_chat_completion(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = 'auto'
    ) -> Iterator[Dict[str, Any]]:
        """Stream chat completion from DeepSeek runtime"""
        # For streaming, we'll simulate chunks or implement proper streaming
        # This is a simplified implementation
        response = self.chat_completion(messages, temperature, max_tokens, tools, tool_choice)
        
        # Simulate streaming by yielding the full response as one chunk
        # In production, this should use proper streaming from the runtime
        yield response
    
    def _format_response(self, raw_response: Dict[str, Any]) -> Dict[str, Any]:
        """Format DeepSeek response to OpenAI-compatible format"""
        return {
            'id': raw_response.get('id', 'deepseek-response'),
            'object': 'chat.completion',
            'created': raw_response.get('created', 0),
            'model': 'deepseek-chat',
            'choices': [
                {
                    'index': 0,
                    'message': raw_response.get('message', {}),
                    'finish_reason': raw_response.get('finish_reason', 'stop')
                }
            ],
            'usage': raw_response.get('usage', {
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'total_tokens': 0
            })
        }
