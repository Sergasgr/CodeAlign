import ast
import re

c_style_comments = ['//', '/*', '*/', '*']
COMMENTS_PER_LANGUAGE = {
    "python": ['#', '"""'],
    "java": c_style_comments,
    "cpp": c_style_comments,
    "c++": c_style_comments,
    "c_sharp": c_style_comments,
    "javascript": c_style_comments,
    "typescript": c_style_comments,
    "go": c_style_comments,
    "rust": c_style_comments
}

NEW_FILE_PROMPT = re.compile(r"Write (\w+) code for the following: ")
EDIT_PROMPT = re.compile(r"Here is the current file:\n```(\w*)\n(.*)\n```\n\nApply this change: ", re.DOTALL)

def language_from_prompt(prompt: str) -> str | None:
    match = EDIT_PROMPT.match(prompt) or NEW_FILE_PROMPT.match(prompt)
    return match.group(1) if match and match.group(1) else None

def filter_comments(code: str, language: str) -> bool: 
    lines = code.split("\n")
    total_comments = 0
    non_blank_lines = 0
    for line in lines:
        if line.strip():
            non_blank_lines += 1
            if any(line.strip().startswith(c) for c in COMMENTS_PER_LANGUAGE[language]):
                total_comments += 1
    return total_comments / non_blank_lines >= 0.9 if non_blank_lines > 0 else False

def filter_no_definitions(code: str) -> bool:
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                                 ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Call)):
                return False
        return True
    except SyntaxError:
        return False

def count_non_comment_lines(code: str, language: str) -> int:
    count = 0
    for line in code.split("\n"):
        if line.strip():
            if not any(line.strip().startswith(c) for c in COMMENTS_PER_LANGUAGE[language]):
                count += 1
    return count

def filter_gutted_edit(code: str, original_prompt: str, language: str) -> bool:
    match = EDIT_PROMPT.match(original_prompt)
    if not match:
        return False
    original_code = match.group(2)
    return count_non_comment_lines(code, language) < 0.3 * count_non_comment_lines(original_code, language)
