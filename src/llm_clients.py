import os
import requests
import time
from typing import Optional

class NIMClient:
    """
    NVIDIA NIM OpenAI-compatible chat client
    """
    def __init__(
        self,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url or os.environ.get(
            "NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"
        )
        self.api_key = api_key or os.environ.get("NVIDIA_API_KEY")

        if not self.api_key:
            raise RuntimeError("NVIDIA_API_KEY not set")

    def generate(self, prompt: str, temperature: Optional[float] = None, max_tokens: Optional[int] = None, max_retries: int = 3) -> str:
        temp = self.temperature if temperature is None else temperature
        mtok = self.max_tokens if max_tokens is None else max_tokens

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": temp,
            "max_tokens": mtok,
        }

        # Retry logic with exponential backoff
        for attempt in range(max_retries):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=90)
                if resp.status_code != 200:
                    error_msg = f"NIM error {resp.status_code}: {resp.text}"
                    # Try to extract more details
                    try:
                        error_data = resp.json()
                        if "detail" in error_data:
                            error_msg = f"NIM error {resp.status_code}: {error_data['detail']}"
                    except:
                        pass
                    raise RuntimeError(error_msg)

                data = resp.json()
                return data["choices"][0]["message"]["content"]
            
            except requests.exceptions.ConnectTimeout as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # exponential backoff: 1s, 2s, 4s
                    print(f"\nConnection timeout (attempt {attempt + 1}/{max_retries}). Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(f"NIM connection failed after {max_retries} attempts: {str(e)}")
            
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"\nRequest failed (attempt {attempt + 1}/{max_retries}). Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(f"NIM request failed after {max_retries} attempts: {str(e)}")
