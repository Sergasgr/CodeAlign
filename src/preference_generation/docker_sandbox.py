""" CODE IF I DECIDE TO CONTINUE WITH THE DAEMON AND NOT WITH THE GRPC
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
"""
import grpc
from src.preference_generation.preference_generation_config import EXECUTION_TIMEOUT

import src.preference_generation.executor_pb2 as executor_pb2
import src.preference_generation.executor_pb2_grpc as executor_pb2_grpc

class DockerSandbox:
    def __init__(self, daemon_url: str = "localhost:3000"):
        self.daemon_url = daemon_url
        self.channel = grpc.insecure_channel(self.daemon_url)
        self.stub = executor_pb2_grpc.ExecutorServiceStub(self.channel)
        
    def execute_code(self, code: str, language: str, timeout_seconds: int = EXECUTION_TIMEOUT) -> dict:
        try:
            request = executor_pb2.ExecutionRequest( # "ExecutionRequest" is not a known attribute of module "src.preference_generation.executor_pb2"
                language=language, 
                code=code
            )
            
            response = self.stub.ExecuteCode(request, timeout=timeout_seconds + 2)
        
            return {
                "success": response.success,
                "stdout": response.stdout,
                "stderr": response.stderr
            }
                
        except grpc.RpcError as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Daemon Error ({e.code().name}): {e.details()}"
            }
            
    def __del__(self):
        if hasattr(self, 'channel'):
            self.channel.close()