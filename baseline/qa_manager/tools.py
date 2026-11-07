from collections import defaultdict
from threading import Lock
import logging
from typing import Dict, List


# per 1K tokens
MODEL_PRICING = {
    'gpt-4o-2024-11-20': {'prompt': 0.00375, 'completion': 0.015},
    'gpt-4o-mini-2024-07-18': {'prompt': 0.000225,'completion': 0.0009},
    'gpt-3.5-turbo-0125': {'prompt': 0.00075,'completion': 0.00225},
    'gpt-4.1-nano-2025-04-14': {'prompt': 0.00015,'completion': 0.0006},
    # Groq models (https://console.groq.com/docs/pricing)
    'llama-3.1-8b-instant': {'prompt': 0.00005, 'completion': 0.00008},
    'llama-3.3-70b-versatile': {'prompt': 0.00059, 'completion': 0.00079},
    # HuggingFace free inference fallback
    'Qwen/Qwen2.5-7B-Instruct': {'prompt': 0.0, 'completion': 0.0},
    'meta-llama/Meta-Llama-3.1-8B-Instruct': {'prompt': 0.0, 'completion': 0.0},
    # OpenRouter free tier (https://openrouter.ai/models?q=free)
    'meta-llama/llama-3.3-70b-instruct:free': {'prompt': 0.0, 'completion': 0.0},
    'meta-llama/llama-3.1-8b-instruct:free': {'prompt': 0.0, 'completion': 0.0},
}

class TokenUsageTracker:
    _instance = None
    _lock = Lock()

    def __new__(cls):
        with cls._lock:
            if not cls._instance:
                cls._instance = super(TokenUsageTracker, cls).__new__(cls)
                cls._instance.logs: List[Dict] = []
            return cls._instance

    def record(self, usage: Dict) -> None:
        model = usage.get('model', '')
        pricing = MODEL_PRICING.get(model, {'prompt': 0.0, 'completion': 0.0})
        completion_pricing = pricing['completion']
        prompt_pricing = pricing['prompt']

        self.logs.append({
            'model': model,
            'agent_name': usage.get('agent_name', ''),
            'completion_tokens': usage.get('completion_tokens', 0),
            'prompt_tokens': usage.get('prompt_tokens', 0),
            'total_tokens': usage.get('total_tokens', 0),
            'completion_price': completion_pricing * usage.get('completion_tokens', 0) / 1000,
            'prompt_price': prompt_pricing * usage.get('prompt_tokens', 0) / 1000,
            'total_price': completion_pricing * usage.get('completion_tokens', 0) / 1000 + prompt_pricing * usage.get('prompt_tokens', 0) / 1000,
        })

    def get_usage(self) -> List[Dict]:
        return [
            {
                'calls': sum([1 for l in self.logs]),
                'completion_tokens': sum([l['completion_tokens'] for l in self.logs]),
                'prompt_tokens': sum([l['prompt_tokens'] for l in self.logs]),
                'total_tokens': sum([l['total_tokens'] for l in self.logs]),
                'completion_price': sum([l['completion_price'] for l in self.logs]),
                'prompt_price': sum([l['prompt_price'] for l in self.logs]),
                'total_price': sum([l['total_price'] for l in self.logs]),
            }
        ]


def setup_logger(name='logger', log_file='log.log', level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)

        formatter = logging.Formatter(
            '%(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


def setup_logger_no_print(name='logger', log_file='log.log', level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)

        formatter = logging.Formatter(
            '%(message)s'
        )
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

    return logger
