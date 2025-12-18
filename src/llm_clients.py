import os
import requests
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

    def generate(self, prompt: str, temperature: Optional[float] = None) -> str:
        temp = self.temperature if temperature is None else temperature

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
            "max_tokens": self.max_tokens,
        }

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
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
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"NIM request failed: {str(e)}")
