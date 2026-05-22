#!/usr/bin/env python3
"""
Tool Call Parser for handling function calling in responses
"""

import json
import re
from typing import Dict, Any, List, Optional, Tuple


def extract_tool_calls(content: str) -> List[Dict[str, Any]]:
    """
    Extract tool/function calls from model response content
    
    Supports formats:
    - <function=name>{"arg": "value"}</function>
    - ```json\n{"name": "function_name", "arguments": {...}}\n```
    - Direct JSON with function call structure
    """
    tool_calls = []
    
    # Pattern 1: XML-style function tags
    xml_pattern = r'<function=(\w+)>(.*?)</function>'
    matches = re.findall(xml_pattern, content, re.DOTALL)
    for match in matches:
        func_name, args_str = match
        try:
            arguments = json.loads(args_str.strip())
            tool_calls.append({
                'type': 'function',
                'function': {
                    'name': func_name,
                    'arguments': json.dumps(arguments)
                }
            })
        except json.JSONDecodeError:
            # If not JSON, treat as plain string argument
            tool_calls.append({
                'type': 'function',
                'function': {
                    'name': func_name,
                    'arguments': args_str.strip()
                }
            })
    
    # Pattern 2: Markdown code block with JSON
    json_pattern = r'```json\s*(\{.*?\})\s*```'
    matches = re.findall(json_pattern, content, re.DOTALL)
    for match in matches:
        try:
            data = json.loads(match)
            if 'name' in data and 'arguments' in data:
                tool_calls.append({
                    'type': 'function',
                    'function': {
                        'name': data['name'],
                        'arguments': json.dumps(data['arguments']) if isinstance(data['arguments'], dict) else str(data['arguments'])
                    }
                })
        except json.JSONDecodeError:
            continue
    
    return tool_calls


def parse_assistant_message(message: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse assistant message to extract tool calls if present
    Returns normalized message with tool_calls field
    """
    if 'tool_calls' in message and message['tool_calls']:
        return message
    
    content = message.get('content', '')
    if not content:
        return message
    
    tool_calls = extract_tool_calls(content)
    if tool_calls:
        result = message.copy()
        result['tool_calls'] = tool_calls
        # Optionally clean up content by removing tool call markup
        cleaned_content = re.sub(r'<function=\w+>.*?</function>', '', content, flags=re.DOTALL)
        cleaned_content = re.sub(r'```json\s*\{.*?\}\s*```', '', cleaned_content, flags=re.DOTALL)
        result['content'] = cleaned_content.strip()
        return result
    
    return message


def build_tool_response_message(
    tool_call_id: str,
    content: str,
    name: Optional[str] = None
) -> Dict[str, Any]:
    """Build a tool response message for conversation"""
    message = {
        'role': 'tool',
        'tool_call_id': tool_call_id,
        'content': content
    }
    if name:
        message['name'] = name
    return message


def format_tool_calls_for_api(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Format tool calls for OpenAI API compatibility"""
    formatted = []
    for i, tc in enumerate(tool_calls):
        formatted.append({
            'id': f"call_{i}",
            'type': tc.get('type', 'function'),
            'function': tc.get('function', {})
        })
    return formatted
