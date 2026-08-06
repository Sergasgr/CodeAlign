import docker
from docker.errors import ContainerError, ImageNotFound, APIError
from src.preference_generation.preference_generation_config import SANDBOX_IMAGE_NAME, EXECUTION_TIMEOUT

class UnsupportedLanguage(Exception):
    pass

def code_command(code: str, language: str, timeout_seconds: int = EXECUTION_TIMEOUT): 
    def run_via_sh(script: str) -> list[str]:
        return ["timeout", str(timeout_seconds), "sh", "-c", script, "sh", code]
    
    match language: # AMPLIACIÓN FASE 7 JUNTO CON LINTERS (docker con toolchains)
        case "python":
            return ["timeout", str(timeout_seconds), "python", "-c", code]
        case "javascript" | "js":
            return ["timeout", str(timeout_seconds), "node", "-e", code]
        case "typescript" | "ts":
            return ["timeout", str(timeout_seconds), "ts-node", "-e", code]
        case "java":
            return run_via_sh(r'printf "%s\n" "$1" > Main.java && javac Main.java && java Main')
        case "cpp" | "c++":
            return run_via_sh(r'printf "%s\n" "$1" > main.cpp && g++ main.cpp -o main && ./main')
        case "c_sharp":
            return run_via_sh(r'printf "%s\n" "$1" > Program.cs && dotnet script Program.cs')
        case "go":
            return run_via_sh(r'printf "%s\n" "$1" > main.go && go run main.go')
        case "rust" | "rs":
            return run_via_sh(r'printf "%s\n" "$1" > main.rs && rustc main.rs && ./main')
        case _:
            raise UnsupportedLanguage(f"Unsupported language: {language}")
               
class DockerSandbox:
    def __init__(self, image_name: str = SANDBOX_IMAGE_NAME):
        self.client = docker.from_env()
        self.image_name = image_name
        
    def execute_code(self, code: str, language: str, timeout_seconds: int = EXECUTION_TIMEOUT) -> dict:
        try:
            command = code_command(code, language, timeout_seconds)
            
            output = self.client.containers.run(
                image=self.image_name,
                command=command,
                remove=True,                 
                network_disabled=True,       
                mem_limit="128m",         
                stderr=True,
                stdout=True
            )
            
            return {
                "success": True,
                "stdout": output.decode("utf-8").strip(),
                "stderr": ""
            }
            
        except ContainerError as e:
            is_timeout = e.exit_status == 124
             
            raw_stderr = e.stderr or ""
            if isinstance(raw_stderr, bytes):
                decoded_stderr = raw_stderr.decode("utf-8", errors="replace")
            else:
                decoded_stderr = str(raw_stderr)
                
            return {
                "success": False,
                "stdout": "",
                "stderr": "Timeout Exceeded" if is_timeout else decoded_stderr.strip()
            }
        except (ImageNotFound, APIError) as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Docker API Error: {str(e)}"
            }
        except UnsupportedLanguage as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Unsupported language: {language}"
            }