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

from src.data_curation.code_smells import check_internal_duplication
from src.data_curation.curation_config import (
    DUPLICATION_DENSITY_THRESHOLD,
    DUPLICATION_WINDOW,
    MAX_COMPLEXITY,
    MAX_LINT_ERRORS,
)
from src.data_curation.linters import ruff_check, cpplint_check, get_cyclomatic_complexity#, eslint_check, pmd_check, clippy_check, golangci_lint_check

LANGUAGES = {  
    "python": Language(tspython.language()),
    "java": Language(tsjava.language()),
    "cpp": Language(tscpp.language()),
    "c++": Language(tscpp.language()),
    "c_sharp": Language(tscsharp.language()),
    "javascript": Language(tsjavascript.language()),
    "typescript": Language(tstypescript.language_typescript()),
    #"go": Language(tsgo.language()),
    #"rust": Language(tsrust.language()),
}

PARSERS = {lang_name: Parser(lang_obj) for lang_name, lang_obj in LANGUAGES.items()}

def validate_syntax(code: str, language: str) -> tuple[bool, str]:
    language = language.lower()
    if language not in PARSERS:
        return False, "Unsupported language"

    tree = PARSERS[language].parse(bytes(code, "utf8")) 
    
    def has_error(node) -> bool:
        if node.type == "ERROR" or node.is_missing:
            return True
        return any(has_error(child) for child in node.children)

    if has_error(tree.root_node):
        return False, "Syntax error detected by tree-sitter"
    return True, ""

# FASE 7 con Docker Image: retomar esta implementación
def linter_check(code: str, language: str): # SEMANTICAL VALIDITY
    match language:
        case "python":
            lint_errors = ruff_check(code)
        case "c++" | "cpp" | "c":
            lint_errors = cpplint_check(code)
        #case "javascript" | "js": 
            #lint_errors = eslint_check(code)
        #case "java":
            #lint_errors = pmd_check(code)
        #case "c_sharp"
        #case "typescript"
        #case "rust" | "rs":
            #lint_errors = clippy_check(code)
        #case "go":
            #lint_errors = golangci_lint_check(code)
        
        case _:
            lint_errors = None 
    return {
        "lint_errors": lint_errors,
        "complexity": get_cyclomatic_complexity(code, language)
    }

def rejection(status: str, error: str) -> dict:
    return {
        "is_valid_syntax": None,
        "lint_errors": None,
        "cyclomatic_complexity": None,
        "flag": False,
        "status": status,
        "error": error,
    }

def check_code(code: str, language: str):
    has_duplication, dup_error = check_internal_duplication(
        code, 
        window_size=DUPLICATION_WINDOW, 
        density_threshold=DUPLICATION_DENSITY_THRESHOLD
    ) # Code smells check
    if has_duplication:
        return rejection("rejected_code_smell", dup_error)
    
    is_valid_ast, syntax_error = validate_syntax(code, language) # CST: Syntactical validity check
    if not is_valid_ast:
        result = rejection("rejected_syntax", syntax_error)
        result["is_valid_syntax"] = False
        return result

    # lint_errors, cycl_complexity = linter_check(code, language).values() # Linters: Semantical validity check
    metrics = linter_check(code, language)
    lint_errors = metrics["lint_errors"]
    cycl_complexity = metrics["complexity"]

    result = {
        "is_valid_ast": is_valid_ast,
        "lint_errors": lint_errors,
        "cyclomatic_complexity": cycl_complexity        
    }
    
    if lint_errors is not None and lint_errors > MAX_LINT_ERRORS:
       result = rejection("rejected_lint", f"Too many lint violations: {lint_errors}")
    elif cycl_complexity > MAX_COMPLEXITY:
        result = rejection("rejected_complexity", f"Cyclomatic complexity too high: {cycl_complexity}")
    else:
        result = {"status": "accepted", "flag": True, "error": None}
        
    result["is_valid_syntax"] = True
    result["lint_errors"] = lint_errors
    result["cyclomatic_complexity"] = cycl_complexity
    return result

def parse_code(code: str) -> str:
    pattern = r"```([a-zA-Z]*)\s*(.*?)```"
    match = re.search(pattern, code, re.DOTALL)
    if match:
        clean_code = match.group(2).strip()
        return clean_code
    return code