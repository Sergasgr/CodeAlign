import httpx
from src.preference_generation.preference_generation_config import EXECUTION_TIMEOUT

class DockerSandbox:
    def __init__(self, daemon_url: str = "http://127.0.0.1:3000/execute"):
        self.daemon_url = daemon_url
        
    def execute_code(self, code: str, language: str, timeout_seconds: int = EXECUTION_TIMEOUT) -> dict:
        try:
            payload = {
                "language": language,
                "code": code
            }
            
            response = httpx.post(
                self.daemon_url, 
                json=payload, 
                timeout=timeout_seconds + 2
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": f"Daemon Error ({response.status_code}): {response.text}"
                }
                
        except httpx.RequestError as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Failed to connect to Rust execution daemon: {str(e)}"
            }