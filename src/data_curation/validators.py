import py_compile #?
import ast
import re
# from radon.complexity import cc_visit, cc_rank
from src.data_curation.linters import ruff_check, cpplint_check, eslint_check, pmd_check, clippy_check, golangci_lint_check, lizard_check, get_cyclomatic_complexity
from src.data_curation.config import MAX_LINT_ERRORS, MAX_COMPLEXITY

"""
import httpx
from httpx import AsyncClient
"""

"""
grades = { #Mover de fichero?? Borrar? Establecer criterio puntuación. Puntuacion 0-1 como ahora o criterios tipo numero de errores y complejidad
    'A': 1,
    'B': 0.75,
    'C': 0.25,
    'D': 0,
    'E': -0.5,
    'F': -1
}
"""

def parse_code(code: str) -> tuple[str, str]:
    pattern = r"```([a-zA-Z]*)\s*(.*?)```"
    match = re.search(pattern, code, re.DOTALL)
    if match:
        language = match.group(1).strip().lower()
        clean_code = match.group(2).strip()
        return language, clean_code
    return "", ""

""" Considerar que hacer -> por el momento opto por los linters de linter.py. Reconsiderar utilizar qodana de otra forma.
async def qodana_check(language: str, code: str):
    QODANA_TOKEN = os.getenv("QODANA_TOKEN", "")
    with httpx.AsyncClient() as client:
        try:
            response = client.get()
        execpt HTTPException 
    #return {"lint_errors": 0, "complexity": 0}
"""

""" RADON
def cyclomatic_complexity(code: str):
    blocks = cc_visit(code)
    complexity = sum(block.complexity for block in blocks)
    #grade = cc_rank(int(complexity))
    #return grades.get(grade)
    cc = cc_visit(f"def _():\n" + "\n".join(f" {l}" for l in codigo.splitlines())).complexity
    return {"linter": "radon", "score": cc_rank(cc), "cc_num": cc, "error": None}
    return complexity
"""

def lintern_check(code: str, language: str): # SEMANTICAL VALIDITY
    # Considerar métricas: extracting metrics like cyclomatic complexity, code duplication, unused variables or naming conventions. Realmente las estoy extrayendo??
    match language:
        case "python":
            lint_errors = ruff_check(code)
        case "c++" | "cpp" | "c":
            lint_errors = cpplint_check(code)
        case "javascript" | "js":
            lint_errors = eslint_check(code)
        case "java":
            lint_errors = pmd_check(code)
        case "rust" | "rs":
            lint_errors = clippy_check(code)
        case "go":
            lint_errors = golangci_lint_check(code)
        case _:
            lint_errors = 0 # Deberia buscar un linter general para que analice cualquier otro lenguaje distinto al especificado?
    return {
        "lint_errors": lint_errors,
        "complexity": get_cyclomatic_complexity(code)
    }
    
def ast_code(code: str) -> bool: # SYNTACTICAL VALIDITY
    """
    if code.startswith("```python"):
        try:
           py_compile.compile(code, doraise=True)
           return True
        except py_compile.PyCompileError as e:
            return False # devolver tambien (linea,fallo)
    else:
    """
    # Actualmente AST solo funcionará si el lenguaje es Python. Si no es Python, asume que la sintaxis es válida (porque lo validaremos con los linters) -> QUIERO una forma para verificar si el codigo en cualquier lenguaje es o no sintacticamente correcto
    
    """
    # Corrección: Solo validamos AST si es Python
    if language == "python":
        is_valid_ast, syntax_error = ast_code(code)
    else:
        is_valid_ast, syntax_error = True, ""
        
    lint_errors, cycl_complexity = lintern_check(code, language).values() 
    # ... resto de la función intacta
    """
    try: 
        ast.parse(code)
        return (True, "")
    except SyntaxError as e:
        return (False, f"Line {e.lineno}: {e.msg}") # e.lineno (linea de error) ; e.msg (motivo del fallo)

def check_code(code: str, language: str):
    is_valid_ast, syntax_error = ast_code(code)
    lint_errors, cycl_complexity = lintern_check(code, language).values() 
    result = {
        "is_valid_ast": is_valid_ast,
        "lint_errors": lint_errors,
        "cyclomatic_complexity": cycl_complexity        
    }
    if not is_valid_ast:
        result["flag"] = False
        result["status"] = "rejected_ast"
        result["error"] = syntax_error 
    elif lint_errors > MAX_LINT_ERRORS:
       result["flag"] = False 
       result["status"] = "rejected_lint"
       result["error"] = f"Too many style/logic errors: {lint_errors}" # Realmente aquí me gustaría alguna forma de que explícitamente dijera el error devuelto por el lintern
    elif cycl_complexity > MAX_COMPLEXITY:
        result["flag"] = False 
        result["status"] = "rejected_complexity"
        result["error"] = f"Code is too complex. Cyclomatic complexity: {cycl_complexity}"
    else:
        result["flag"] = True
        result["status"] = "accepted"
    return result