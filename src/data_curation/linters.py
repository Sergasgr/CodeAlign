import tempfile 
import os
import subprocess
import lizard
import threading
from src.data_curation.curation_config import PMD_TIMEOUT_SECONDS, CLIPPY_TIMEOUT_SECONDS, DOTNET_TIMEOUT_SECONDS, RUFF_TIMEOUT_SECONDS, CPPLINT_TIMEOUT_SECONDS, ESLINT_TIMEOUT_SECONDS, GOLANG_TIMEOUT_SECONDS

# CSHARP_PROJ_DIR = None
csharp_local = threading.local()

LIZARD_EXTENSION = {
    "python": ".py",
    "java": ".java",
    "cpp": ".cpp",
    "c_sharp": ".cs",
    "javascript": ".js",
    "typescript": ".ts",
    "go": ".go",
    "rust": ".rs",
}

def ruff_check(code: str) -> int: 
    with tempfile.NamedTemporaryFile(suffix=".py", delete=True) as tmp:
        tmp.write(code.encode('utf-8'))
        tmp.flush()
        try:
            result = subprocess.run(
                ["ruff", "check", "--select=F,E9", "--output-format=concise", tmp.name], 
                capture_output=True, 
                text=True,
                timeout=RUFF_TIMEOUT_SECONDS
            )
        except subprocess.TimeoutExpired:
            return -1 
    out = result.stdout.strip()
    return len(out.split("\n")) if out else 0

def cpplint_check(code: str) -> int:
    with tempfile.NamedTemporaryFile(suffix=".cpp", delete=True) as tmp:
        tmp.write(code.encode('utf-8'))
        tmp.flush()
        try:
            result = subprocess.run(
                ["cpplint",
                 "--filter=-legal,-build/header_guard,-build/include_order,"
                 "-build/namespaces,-whitespace,-readability/casting",
                 "--counting=total", tmp.name], 
                capture_output=True, 
                text=True, 
                check=False,
                timeout=CPPLINT_TIMEOUT_SECONDS
            )
        except subprocess.TimeoutExpired:
            return -1
    # cpplint writes to stderr; last line is "Total errors found: N"
    for line in reversed(result.stderr.splitlines()):
        if "Total errors found:" in line:
            try:
                return int(line.split(":")[-1].strip())
            except ValueError:
                break
    return 0

def eslint_check(code: str) -> int:
    rules_json = '{"no-undef": "error", "no-unused-vars": "warn", "no-redeclare": "error", "no-dupe-keys": "error", "no-unreachable": "error", "no-constant-condition": "error", "no-empty": "warn", "valid-typeof": "error"}'
    with tempfile.NamedTemporaryFile(suffix=".js", delete=True) as tmp:
        tmp.write(code.encode('utf-8'))
        tmp.flush()
        try:
            result = subprocess.run(
                ["eslint", "--no-eslintrc", "--rule", rules_json, "--format", "compact", tmp.name],
                capture_output=True, 
                text=True, 
                check=False,
                timeout=ESLINT_TIMEOUT_SECONDS
            )
        except subprocess.TimeoutExpired:
            return -1
    out = result.stdout.strip()
    if not out:
        return 0
    return len([line for line in out.split('\n') if "Error" in line or "Warning" in line])

def pmd_check(code: str) -> int: #considerar si pmd o checkstyle
    with tempfile.NamedTemporaryFile(suffix=".java", delete=True) as tmp:
        tmp.write(code.encode('utf-8'))
        tmp.flush()
        try:
            result = subprocess.run(
                ["pmd", "check", "-f", "text", "-R", "category/java/errorprone.xml,category/java/bestpractices.xml", "-d", tmp.name],
                capture_output=True, text=True, check=False, timeout=PMD_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            return -1  # infra timeout, no confundir con conteo real de errores
    out = result.stdout.strip()
    if not out or "No problems found" in out:
        return 0
    return len([line for line in out.split('\n') if tmp.name in line])

def clippy_check(code: str) -> int:
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "snippet.rs")
        with open(src_path, "w", encoding="utf-8") as f:
            f.write(code)
        try:
            result = subprocess.run(
                ["clippy-driver", "--edition", "2021", "--emit=metadata", "--out-dir", tmpdir, src_path],
                capture_output=True, text=True, check=False, timeout=CLIPPY_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            return -1
    out = result.stderr.strip()
    if not out:
        return 0
    # Only count clippy warnings (code quality lint), not compilation errors.
    # Compilation errors on standalone snippets are missing-crate issues, not quality signals.
    return len([line for line in out.split('\n') if line.startswith("warning:")])

def golangci_lint_check(code: str) -> int:
    with tempfile.NamedTemporaryFile(suffix=".go", delete=True) as tmp:
        tmp.write(code.encode('utf-8'))
        tmp.flush()
        try:
            result = subprocess.run(
                ["golangci-lint", "run",
                 "--enable=govet,errcheck,staticcheck",
                 "--disable-all",
                 "--out-format=line-number", tmp.name],
                capture_output=True, text=True, check=False,
                timeout=GOLANG_TIMEOUT_SECONDS
            )
        except subprocess.TimeoutExpired:
            return -1
    out = result.stdout.strip()
    if not out:
        return 0
    return len(out.split('\n'))

def tsc_check(code: str) -> int:
    """Lint TypeScript with eslint (semantic rules only, no type-checking).

    tsc --noEmit performs full type resolution and rejects any snippet with
    unresolved imports/types — that's not a code quality signal for standalone
    snippets. Using eslint with the same semantic rules as JavaScript gives a
    fair quality gate.
    """
    rules_json = '{"no-undef": "error", "no-unused-vars": "warn", "no-redeclare": "error", "no-dupe-keys": "error", "no-unreachable": "error", "no-constant-condition": "error", "no-empty": "warn", "valid-typeof": "error"}'
    with tempfile.NamedTemporaryFile(suffix=".ts", delete=True) as tmp:
        tmp.write(code.encode('utf-8'))
        tmp.flush()
        try:
            result = subprocess.run(
                ["eslint", "--no-eslintrc", "--rule", rules_json,
                 "--format", "compact", tmp.name],
                capture_output=True,
                text=True,
                check=False,
                timeout=ESLINT_TIMEOUT_SECONDS
            )
        except subprocess.TimeoutExpired:
            return -1
    out = result.stdout.strip()
    if not out:
        return 0
    return len([line for line in out.split('\n') if "Error" in line or "Warning" in line])

def get_csharp_project() -> str:
    if not hasattr(csharp_local, "proj_dir"):
        tmpdir = tempfile.mkdtemp(prefix="codealign_csharp_")
        try:
            subprocess.run(
                ["dotnet", "new", "console", "-n", "TempProj"],
                cwd=tmpdir, 
                capture_output=True, 
                check=False,
                timeout=DOTNET_TIMEOUT_SECONDS
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("C# Project initialization timed out. Network issue or dead lock.")
        csharp_local.proj_dir = os.path.join(tmpdir, "TempProj")
    return csharp_local.proj_dir

def csharp_check(code: str) -> int:
    proj_dir = get_csharp_project()
    program_path = os.path.join(proj_dir, "Program.cs")
    
    with open(program_path, "w", encoding="utf-8") as f:
        f.write(code)
        
    try:
        result = subprocess.run(
            ["dotnet", "build", "--nologo", "-clp:NoSummary"],
            cwd=proj_dir, 
            capture_output=True, 
            text=True, 
            check=False,
            timeout=DOTNET_TIMEOUT_SECONDS 
        )
    except subprocess.TimeoutExpired:
        return -1 # Infra timeout
    
    out = result.stdout.strip()
    if not out:
        return 0
    # Only count warnings (code quality signals like CS0219 unused variable),
    # not compilation errors (CS0246 missing namespace — snippet context issue).
    return len([line for line in out.split('\n') if " warning CS" in line])

def get_cyclomatic_complexity(code: str, language: str) -> int:
    ext = LIZARD_EXTENSION.get(language, ".txt")
    info = lizard.analyze_file.analyze_source_code(f"snippet{ext}", code)
    if not info.function_list:
        return 0
    return max(func.cyclomatic_complexity for func in info.function_list)