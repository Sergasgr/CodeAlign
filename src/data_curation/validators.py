#import ast

import re 
from tree_sitter import Language, Parser
import tree_sitter_python as tspython
import tree_sitter_java as tsjava
import tree_sitter_cpp as tscpp
import tree_sitter_c_sharp as tscsharp
import tree_sitter_javascript as tsjavascript
import tree_sitter_typescript as tstypescript
import tree_sitter_go as tsgo
import tree_sitter_rust as tsrust

# from radon.complexity import cc_visit, cc_rank
from src.data_curation.linters import ruff_check, cpplint_check, eslint_check, pmd_check, clippy_check, golangci_lint_check, get_cyclomatic_complexity
from src.data_curation.code_smells import check_internal_duplication
from configs.config import MAX_LINT_ERRORS, MAX_COMPLEXITY 
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

# mover diccionario / implementacion de aqui de aquí?? -> hacer otro .py con languages y parsers?
languages = {  
    "python": Language(tspython.language()),
    "java": Language(tsjava.language()),
    "cpp": Language(tscpp.language()),
    "c++": Language(tscpp.language()),
    "c_sharp": Language(tscsharp.language()),
    "javascript": Language(tsjavascript.language()),
    "typescript": Language(tstypescript.language()), # "language" is not a known attribute of module "tree_sitter_typescript"
    "go": Language(tsgo.language()),
    "rust": Language(tsrust.language()),
}

parsers = {}
for lang_name, lang_obj in languages.items():
    parser = Parser(lang_obj)
    parsers[lang_name] = parser

def parse_code(text: str) -> tuple[str, str]:
    pattern = re.compile(r"```([a-zA-Z]*)\s*(.*?)```", re.DOTALL)
    match = pattern.search(text)
    if not match:
        return "", ""
    language = match.group(1).strip().lower()
    code = match.group(2).strip()
    return language, code

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

# def ast_code(code: str) -> tuple[bool, str]: # SYNTACTICAL VALIDITY
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
"""
    try:
            ast.parse(code)
            return True, ""
        except SyntaxError as e:
            return False, f"Line {e.lineno}: {e.msg}" # e.lineno (linea de error) ; e.msg (motivo del fallo)
"""

def validate_syntax(code: str, lang: str) -> tuple[bool, str]:
    if lang.lower() not in parsers:
        return False, "Unsupported language"
    tree = parsers[lang].parse(bytes(code, "utf8"))  

    def dfs(node):
        if node.type == 'ERROR' or node.is_missing:
            return False
        for child in node.children:
            if not dfs(child):
                return False
        return True

    is_valid = dfs(tree.root_node)
    if not is_valid:
        return False, "Syntax error detected by tree-sitter"

    return True, ""

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

def check_code(code: str, language: str):
    has_duplication, dup_error = check_internal_duplication(code, window_size=4) # Code smells check
    if has_duplication:
        return {
            "flag": False,
            "status": "rejected_code_smell",
            "error": dup_error,
            "is_valid_ast": False, 
            "lint_errors": 0,
            "cyclomatic_complexity": 0
        }
    is_valid_ast, syntax_error = ast_code(code) # AST: Syntactical validity check
    if not is_valid_ast:
        return {
            "flag": False,
            "status": "rejected_ast",
            "error": syntax_error,
            "is_valid_ast": False,
            "lint_errors": 0,
            "cyclomatic_complexity": 0
        }
    lint_errors, cycl_complexity = lintern_check(code, language).values() # Linters: Semantical validity check
    result = {
        "is_valid_ast": is_valid_ast,
        "lint_errors": lint_errors,
        "cyclomatic_complexity": cycl_complexity        
    }
    if lint_errors > MAX_LINT_ERRORS:
       result["flag"] = False 
       result["status"] = "rejected_lint"
       result["error"] = f"Too many style/logic errors: {lint_errors}"
    elif cycl_complexity > MAX_COMPLEXITY:
        result["flag"] = False 
        result["status"] = "rejected_complexity"
        result["error"] = f"Code is too complex. Cyclomatic complexity: {cycl_complexity}"
    else:
        result["flag"] = True
        result["status"] = "accepted"
    return result