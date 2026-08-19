/* Incluir en README.md????
La librería tonic-build de Rust no compila los archivos .proto por sí sola; actúa como un puente hacia el compilador oficial de Google instalado en tu sistema. Si no lo tienes, fallará.  Instálalo en tu máquina local (donde estás ejecutando Rust):Bashsudo apt update
sudo apt install protobuf-compiler
(Puedes verificar que se ha instalado correctamente ejecutando protoc --version en tu terminal).

Incluir en ejecucion junto con scripts??
cargo clean
cargo run


Acción A (Timeout doble): En rust-daemon/src/main.rs (aprox. líneas 45-46), el timeout externo (tokio) y el interno (bash) son ambos de 5 segundos. Cambia el timeout de tokio a 15 segundos para darle margen a Docker: let limit = Duration::from_secs(15);
Acción B (dotnet script): En el mismo archivo (línea 144), intentas usar dotnet script pero no está instalado en la imagen de Docker. Añade la instalación al Dockerfile.executor: RUN dotnet tool install -g dotnet-script (y asegúrate de que el PATH incluya las tools globales de dotnet).
*/

use std::time::Duration;
use tokio::process::Command;
use tokio::time::timeout;
use tonic::{Request, Response, Status, transport::Server};

pub mod codealign {
    tonic::include_proto!("codealign"); //`OUT_DIR` not set, build scripts may have failed to runrust-analyzermacro-error
}

use codealign::executor_service_server::{ExecutorService, ExecutorServiceServer};
use codealign::{ExecutionRequest, ExecutionResponse};

#[derive(Debug, Default)]
pub struct MyExecutor;

/*
lifetime parameters or bounds on method `execute_code` do not match the trait declaration
lifetimes do not match method in trait
codealign.rs(148, 18): lifetimes in impl do not match this method in trait
codealign.rs(149, 13): this bound might be missing in the impl
codealign.rs(146, 5): this bound might be missing in the impl
*/

impl ExecutorService for MyExecutor {
    async fn execute_code(
        &self,
        request: Request<ExecutionRequest>,
    ) -> Result<Response<ExecutionResponse>, Status> {
        let req_data = request.into_inner();
        let language = req_data.language;
        let code = req_data.code;

        println!("Executing code in: {}", language);

        let limit = Duration::from_secs(5);

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

/*
Dos cosas de diseño que conviene revisar después de que compile

El timeout externo se come al interno. Envuelves command.output() en un tokio::time::timeout(Duration::from_secs(5), ...), y dentro del contenedor vuelves a aplicar timeout 5 al comando real. Como el docker run en sí (crear el contenedor, cgroups, arrancar el proceso) ya consume parte de esos 5s antes de que el código empiece a correr, el timeout externo casi siempre va a disparar primero — nunca vas a ver el comportamiento "gracioso" del timeout interno, y código que legítimamente terminaría en 5s puede morir por puro overhead de arranque del contenedor. Dale más margen al externo que al interno, por ejemplo 5s dentro / 15s fuera, y ajusta empíricamente.

dotnet script probablemente no existe en codealign-executor:latest. Si esta es la misma imagen del Dockerfile que revisamos antes (mismo tag), instala dotnet-sdk-8.0 pero nunca corre dotnet tool install -g dotnet-script — así que la rama de C# en build_command va a fallar en runtime con "comando no encontrado", no por un bug en el Rust sino porque le falta esa herramienta a la imagen.
*/
