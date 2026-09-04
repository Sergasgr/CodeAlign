// Deprecated
use axum::{
    http::StatusCode,
    routing::post,
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use tokio::process::Command;
use tokio::time::{timeout, Duration};

#[derive(Serialize, Deserialize)]
struct ExecutionRequest {
    language: String,
    code: String,
}

#[derive(Serialize, Deserialize)]
struct ExecutionResponse {
    success: bool,
    stdout: String,
    stderr: String,
}

#[tokio::main]
async fn main() {
    let app = Router::new().route("/execute", post(execution));
    let addr = SocketAddr::from(([127, 0, 0, 1], 3000)); 
    println!("Rust Daemon listening to {}", addr);
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn execution(Json(request): Json<ExecutionRequest>) -> Result<Json<ExecutionResponse>, (StatusCode, String)> {
    let limit = Duration::from_secs(5);
    let args = match build_command(&request.language, &request.code) {
        Ok(args) => args,
        Err(e) => return Err((StatusCode::BAD_REQUEST, e)),
    };

    let mut command = Command::new("docker");
    command.arg("run")
        .arg("--rm")
        .arg("--network=none")
        .arg("--memory=128m")
        .arg("codealign-executor:latest") 
        .args(args);
        
    let output_timeout = timeout(limit, command.output()).await;

     let output_result = match timeout_result {
        Ok(res) => res,
        Err(_) => {
            return Err((
                StatusCode::GATEWAY_TIMEOUT,
                "Error: The code take too long to execute".to_string()
            ));
        }
    };

    let output = match output_result {
        Ok(out) => out,
        Err(err) => {
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Critical error invoking Docker CLI: {}", err)
            ));
        }
    };

    let success = output.status.success();
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();

    Ok(Json(ExecutionResponse {
        success,
        stdout,
        stderr,
    }))
}

fn build_command(language: &str, code: &str) -> Result<Vec<String>, String> { 
    let timeout = "5";
    let run_via_sh = |script: &str| -> Vec<String> {
        vec![
            "timeout".to_string(), 
            timeout.to_string(),
            "sh".to_string(), 
            "-c".to_string(), 
            script.to_string(),
            "sh".to_string(), 
            code.to_string()
        ]
    };

    match language.to_lowercase().as_str() {
        "python" => Ok(vec![
            "timeout".to_string(), 
            timeout.to_string(), 
            "python".to_string(), 
            "-c".to_string(), 
            code.to_string()
        ]),
        "javascript" | "js" => Ok(vec![
            "timeout".to_string(), 
            timeout.to_string(), 
            "node".to_string(), 
            "-e".to_string(), 
            code.to_string()
        ]),
        "typescript" | "ts" => Ok(vec![
            "timeout".to_string(), 
            timeout.to_string(), 
            "ts-node".to_string(), 
            "-e".to_string(), 
            code.to_string()
        ]),
        "java" => Ok(run_via_sh(
            "printf \"%s\\n\" \"$1\" > Main.java && javac Main.java && java Main"
        )),
        "cpp" | "c++" => Ok(run_via_sh(
            "printf \"%s\\n\" \"$1\" > main.cpp && g++ main.cpp -o main && ./main"
        )),
        "c_sharp" => Ok(run_via_sh(
            "printf \"%s\\n\" \"$1\" > Program.cs && dotnet script Program.cs"
        )),
        "go" => Ok(run_via_sh(
            "printf \"%s\\n\" \"$1\" > main.go && go run main.go"
        )),
        "rust" | "rs" => Ok(run_via_sh(
            "printf \"%s\\n\" \"$1\" > main.rs && rustc main.rs && ./main"
        )),
        _ => Err(format!("Unsupported language: {}", language)),
    }
}

