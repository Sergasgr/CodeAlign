use std::time::Duration;
use tokio::process::Command;
use tokio::time::timeout;
use tonic::{Request, Response, Status, transport::Server};

pub mod codealign {
    tonic::include_proto!("codealign");
}

use codealign::executor_service_server::{ExecutorService, ExecutorServiceServer};
use codealign::{ExecutionRequest, ExecutionResponse};

#[derive(Debug, Default)]
pub struct MyExecutor;

#[tonic::async_trait]
impl ExecutorService for MyExecutor {
    async fn execute_code(
        &self,
        request: Request<ExecutionRequest>,
    ) -> Result<Response<ExecutionResponse>, Status> {
        let req_data = request.into_inner();
        let language = req_data.language;
        let code = req_data.code;

        println!("Executing code in: {}", language);

        let limit = Duration::from_secs(15);

        let args = match build_command(&language, &code) {
            Ok(args) => args,
            Err(e) => return Err(Status::invalid_argument(e)),
        };

        let mut command = Command::new("docker");
        command
            .arg("run")
            .arg("--rm")
            .arg("--network=none")
            .arg("--memory=128m")
            .arg("codealign-executor:latest")
            .args(args);

        let output_timeout = timeout(limit, command.output()).await;

        let output_result = match output_timeout {
            Ok(res) => res,
            Err(_) => {
                return Err(Status::deadline_exceeded(
                    "Error: The code took too long to execute",
                ));
            }
        };

        let output = match output_result {
            Ok(out) => out,
            Err(err) => {
                return Err(Status::internal(format!(
                    "Critical error invoking Docker CLI: {}",
                    err
                )));
            }
        };

        let success = output.status.success();
        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        let reply = ExecutionResponse {
            success,
            stdout,
            stderr,
        };

        Ok(Response::new(reply))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr = "[::1]:3000".parse()?;
    let executor = MyExecutor::default();

    println!("gRPC Server listening on {}", addr);

    Server::builder()
        .add_service(ExecutorServiceServer::new(executor))
        .serve(addr)
        .await?;

    Ok(())
}

fn build_command(language: &str, code: &str) -> Result<Vec<String>, String> {
    let timeout_val = "5";
    let run_via_sh = |script: &str| -> Vec<String> {
        vec![
            "timeout".to_string(),
            timeout_val.to_string(),
            "sh".to_string(),
            "-c".to_string(),
            script.to_string(),
            "sh".to_string(),
            code.to_string(),
        ]
    };

    match language.to_lowercase().as_str() {
        "python" => Ok(vec![
            "timeout".to_string(),
            timeout_val.to_string(),
            "python".to_string(),
            "-c".to_string(),
            code.to_string(),
        ]),
        "javascript" | "js" => Ok(vec![
            "timeout".to_string(),
            timeout_val.to_string(),
            "node".to_string(),
            "-e".to_string(),
            code.to_string(),
        ]),
        "typescript" | "ts" => Ok(vec![
            "timeout".to_string(),
            timeout_val.to_string(),
            "ts-node".to_string(),
            "-e".to_string(),
            code.to_string(),
        ]),
        "java" => Ok(run_via_sh(
            "printf \"%s\\n\" \"$1\" > Main.java && javac Main.java && java Main",
        )),
        "cpp" | "c++" => Ok(run_via_sh(
            "printf \"%s\\n\" \"$1\" > main.cpp && g++ main.cpp -o main && ./main",
        )),
        "c_sharp" => Ok(run_via_sh(
            "printf \"%s\\n\" \"$1\" > Program.cs && dotnet script Program.cs",
        )),
        "go" => Ok(run_via_sh(
            "printf \"%s\\n\" \"$1\" > main.go && go run main.go",
        )),
        "rust" | "rs" => Ok(run_via_sh(
            "printf \"%s\\n\" \"$1\" > main.rs && rustc main.rs && ./main",
        )),
        _ => Err(format!("Unsupported language: {}", language)),
    }
}
