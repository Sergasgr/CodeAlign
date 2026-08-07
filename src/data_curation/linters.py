import tempfile 
import os
import subprocess
import lizard

LIZARD_EXTENSION = {
    "python": ".py",
    "java": ".java",
    "cpp": ".cpp",
    "c_sharp": ".cs",
    "javascript": ".js",
    "typescript": ".ts",
}

def ruff_check(code: str) -> int: 
    with tempfile.NamedTemporaryFile(suffix=".py", delete=True) as tmp:
        tmp.write(code.encode('utf-8'))
        tmp.flush()
        result = subprocess.run(
            ["ruff", "check", "--select=E,W,F,N", "--output-format=concise", tmp.name], 
            capture_output=True, 
            text=True
        )
    out = result.stdout.strip()
    return len(out.split("\n")) if out else 0

def cpplint_check(code: str) -> int:
    with tempfile.NamedTemporaryFile(suffix=".cpp", delete=True) as tmp:
        tmp.write(code.encode('utf-8'))
        tmp.flush()
        result = subprocess.run(
            ["cpplint", "--countdown", tmp.name], 
            capture_output=True, 
            text=True, 
            check=False
        )
    return len([
        line for line in result.stderr.splitlines() 
        if "Artifact" not in line and "Total errors found" not in line and line.strip()
    ])

def eslint_check(code: str) -> int:
    with tempfile.NamedTemporaryFile(suffix=".js", delete=True) as tmp:
        tmp.write(code.encode('utf-8'))
        tmp.flush()
        result = subprocess.run(
            ["eslint", "--no-eslintrc", "--format", "compact", tmp.name],
            capture_output=True, text=True, check=False
        )
    out = result.stdout.strip()
    if not out:
        return 0
    return len([line for line in out.split('\n') if "Error" in line or "Warning" in line])

def pmd_check(code: str) -> int: #considerar si pmd o checkstyle
    with tempfile.NamedTemporaryFile(suffix=".java", delete=True) as tmp:
        tmp.write(code.encode('utf-8'))
        tmp.flush()
        result = subprocess.run(
            ["pmd", "check", "-f", "text", "-R", "category/java/errorprone.xml", "-d", tmp.name],
            capture_output=True, text=True, check=False
        )
    out = result.stdout.strip()
    if not out or "No problems found" in out:
        return 0
    return len([line for line in out.split('\n') if tmp.name in line])

def clippy_check(code: str) -> int:
    with tempfile.NamedTemporaryFile(suffix=".rs", delete=True) as tmp:
        tmp.write(code.encode('utf-8'))
        tmp.flush()
        result = subprocess.run(
            ["clippy-driver", tmp.name],
            capture_output=True, text=True, check=False
        )
    out = result.stderr.strip()
    if not out:
        return 0
    return len([line for line in out.split('\n') if line.startswith("error:") or line.startswith("warning:")])

def golangci_lint_check(code: str) -> int: #considerar si golangci-lint o revive
    with tempfile.NamedTemporaryFile(suffix=".go", delete=True) as tmp:
        tmp.write(code.encode('utf-8'))
        tmp.flush()
        result = subprocess.run(
            ["golangci-lint", "run", "--out-format=line-number", tmp.name],
            capture_output=True, text=True, check=False
        )
    out = result.stdout.strip()
    if not out:
        return 0
    return len(out.split('\n'))

def tsc_check(code: str) -> int:
    with tempfile.NamedTemporaryFile(suffix=".ts", delete=True) as tmp:
        tmp.write(code.encode('utf-8'))
        tmp.flush()
        result = subprocess.run(
            ["tsc", "--noEmit", tmp.name],
            capture_output=True, text=True, check=False
        )
    out = result.stdout.strip()
    if not out:
        return 0
    return len([line for line in out.split('\n') if "error TS" in line])

def csharp_check(code: str) -> int:
    with tempfile.TemporaryDirectory() as tmpdir:
        subprocess.run(
            ["dotnet", "new", "console", "-n", "TempProj"],
            cwd=tmpdir, capture_output=True, check=False
        )
        proj_dir = os.path.join(tmpdir, "TempProj")
        program_path = os.path.join(proj_dir, "Program.cs")
        with open(program_path, "w", encoding="utf-8") as f:
            f.write(code)
        result = subprocess.run(
            ["dotnet", "build", "--nologo", "-clp:NoSummary"],
            cwd=proj_dir, capture_output=True, text=True, check=False
        )
    out = result.stdout.strip()
    if not out:
        return 0
    return len([line for line in out.split('\n') if " warning CS" in line or " error CS" in line])

def get_cyclomatic_complexity(code: str, language: str) -> int:
    ext = LIZARD_EXTENSION.get(language, ".txt")
    info = lizard.analyze_source_code(f"snippet{ext}", code) # type: ignore
    if not info.function_list:
        return 0
    return max(func.cyclomatic_complexity for func in info.function_list)