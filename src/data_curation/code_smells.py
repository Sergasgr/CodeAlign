def check_internal_duplication(code: str, window_size: int = 4) -> tuple[bool, str]: 
    meaningful_lines = []
    for line in code.split("\n"):
        stripped = line.strip()
        if stripped and stripped not in ('{', '}', '(', ')', '[', ']', ';'):
            meaningful_lines.append(stripped)
    if len(meaningful_lines) < window_size * 2:
        return False, ""
    blocks = set()
    for i in range(len(meaningful_lines) - window_size + 1):
        block = tuple(meaningful_lines[i : i + window_size])
        if block in blocks:
            error_msg = f"Internal duplication detected: {window_size} lines repeated."
            return True, error_msg
        blocks.add(block)
    return False, ""

"""
- Observación: diferenciar entre repetición estructural (necesaria) y repetición semántica (el verdadero code smell). Si pones una ventana de 3 o 4 líneas, te vas a cargar medio dataset. Eso no es código espagueti ni una violación del principio DRY (Don't Repeat Yourself); son cláusulas de guarda (guard clauses) o manejo de errores estándar. Penalizar a un modelo por escribir eso sería un error garrafal, porque es código idiomático y seguro.
- Análisis:
¿Cómo valoramos realmente un "Code Smell" por duplicación?

En la industria (por ejemplo, herramientas profesionales como SonarQube o el Copy/Paste Detector de PMD), un bloque repetido solo se considera un smell si cumple ciertos criterios de "peso". Para tu función check_internal_duplication, aquí tienes las estrategias para que sea un filtro inteligente y no una guillotina ciega:

1. El tamaño de la ventana (Window Size) debe ser sustancial
Un copy-paste dañino (aquel que debió ser abstraído en una función de ayuda) rara vez tiene 3 líneas. Suele ser un bloque lógico completo: la configuración de una conexión, un bucle de transformación de datos complejo, etc.

La solución: Subir el window_size a un mínimo de 10 a 15 líneas de código "limpio" (sin contar llaves ni blancos). Si un desarrollador repite 10 líneas exactas en el mismo archivo, definitivamente debió hacer un refactor.

2. Filtrado de líneas "basura" o comunes
Tu limpieza inicial quitaba llaves y paréntesis, lo cual está muy bien. Pero para evitar falsos positivos con estructuras de control básicas, debes ampliar tu lista negra de líneas ignoradas.

La solución: Antes de añadir una línea a tu lista de meaningful_lines, ignora (haz skip) de líneas que sean exactamente return, return True, return False, break, continue, pass, o que solo contengan sentencias else:.

3. Densidad vs. Aparición (Opcional, más avanzado)
A veces, un archivo de 500 líneas tiene un bloque de 10 líneas repetido por cuestiones de arquitectura, y el 98% restante del código es brillante.

La solución: En lugar de devolver True a la primera coincidencia, puedes contar cuántas líneas caen dentro de bloques duplicados y dividirlo por el total de líneas. Si la duplicación representa más de un 15-20% del código total, lo rechazas.
"""