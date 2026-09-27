import re

IGNORED_LINES = {
    "{", "}", "(", ")", "[", "]", ";",
    "},", "];", ");", "};",
 
    "break", "break;", "continue", "continue;", "pass", "else", "else:",
    "try", "try:", "try {", "finally", "finally:", "finally {",
 
    "return", "return;", "return true", "return true;", "return false", "return false;",
    "return 0", "return 0;", "return 1", "return 1;", "return null", "return null;",
    "return none", "return undefined", "return undefined;", "return nullptr", "return nullptr;",
 
    "public:", "private:", "protected:",
    "default:",
    "except:",
 
    "if err != nil {", "return err", "return err;", "return nil", "return nil;",
    "defer", "panic(err)",
 
    "ok(())", "ok(())?", "err(e)", "_ => {}", "none => {}",
}
CATCH_PATTERN = re.compile(r"^catch\s*\(.*\)\s*\{?$", re.IGNORECASE)

def check_internal_duplication(code: str, window_size: int = 12, density_threshold: float = 0.2) -> tuple[bool, str]:
    meaningful_lines = []
    for line in code.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower() in IGNORED_LINES:
            continue
        if CATCH_PATTERN.match(stripped):
            continue
        meaningful_lines.append(stripped)
        
    if len(meaningful_lines) < window_size * 2:
        return False, ""
    
    seen: set[tuple[str, ...]] = set()
    duplicated_line_count = 0
    for i in range(len(meaningful_lines) - window_size + 1):
        block = tuple(meaningful_lines[i : i + window_size])
        if block in seen:
            duplicated_line_count += window_size
        seen.add(block)
        
    density = duplicated_line_count / len(meaningful_lines)
    if density > density_threshold:
        return True, (
            f"Internal duplication: ~{density:.0%} of the file is inside "
            f"repeated {window_size}+ line blocks"
        )
    return False, ""