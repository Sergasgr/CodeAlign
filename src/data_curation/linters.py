import tempfile 
import subprocess
import lizard

def ruff_check(code: str) -> int: # Score: [0-1]? Establecer criterio de puntuación
    with tempfile.NamedTemporaryFile(suffix=".py", delete=True) as tmp:
        tmp.write(code.encode('utf-8'))
        tmp.flush()
        result = subprocess.run(
            ["ruff", "check", "--select=E,W,F,N", tmp.name], 
            capture_output=True, text=True
        )
    out = result.stdout.strip()
    if not out:
        return 0
    return len(out.split('\n'))
    # SCORE = score = max(0, 1 - (errores * 0.1))?

def cpplint_check(code: str) -> int:
    with tempfile.NamedTemporaryFile(suffix=".cpp", delete=True) as tmp:
        tmp.write(code.encode('utf-8'))
        tmp.flush()
        result = subprocess.run(
            ["cpplint", "--countdown", tmp.name], 
            capture_output=True, text=True, check=False
        )
    return len([line for line in result.stderr.splitlines() if "Artifact" not in line and "Total errors found" not in line and line.strip()])

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

def get_cyclomatic_complexity(code: str) -> int:
    info = lizard.analyze_source_code("snippet.txt", code)
    if not info.function_list:
        return 0
    return max(func.cyclomatic_complexity for func in info.function_list)