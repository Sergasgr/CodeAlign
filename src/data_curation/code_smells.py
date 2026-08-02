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

def check_internal_duplication( 
    code: str,
    window_size: int = 12,
    density_threshold: float = 0.2,
) -> tuple[bool, str]:
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

"""
¿Es realmente una buena forma de prevenir code smells y DRY?

Honestamente: es una barrera rápida y razonable para v1, pero no un detector de clones de verdad — y vale la pena que sepas exactamente dónde está el límite en vez de sobrevenderlo:

Solo detecta copias exactas (Type-1). Un bloque copiado y pegado con una sola variable renombrada ya no hace match — y copy-paste-y-renombrar es, en la práctica, el patrón de duplicación más común, más que la copia byte a byte. Detectar eso necesitaría normalizar identificadores antes de comparar (una pasada "ciega a nombres de variable"), que esto no hace.
Solo detecta bloques contiguos. Si la lógica duplicada está repartida con código distinto en medio, no la pilla.
El comentario que ya tenías sobre el doble conteo de ventanas solapadas sigue siendo válido — es una aproximación, no una métrica de precisión.

Mi recomendación: déjalo así para la Fase 1 (rápido, sin dependencias, cero falsos positivos añadidos), y si el reporte de la Fase 1 muestra que se te está colando duplicación real tipo copy-paste-renombrado, ahí sí merece la pena añadir la normalización de identificadores — como mejora dirigida por evidencia, no especulativa.
"""