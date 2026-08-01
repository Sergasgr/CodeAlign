"""
deprecated -> la duplicación que me interesa penalizar es la repetición de código (Copy-Paste / Code Smells) dentro de un mismo fragmento generado, no si el snippet se parece a otro del dataset. 
"""

import uuid
from datasketch import MinHash, MinHashLSH

def tokenize_code(code: str):
    return code.lower().replace("\n", "").split() 
    # n-grams??
    """
    code = code.lower().replace("\n", "").split() 
    tokens = []
    for i in range(len(code)-1): 3-gramas; añadir $??
        tokens.append((code[i], code[i+1], code[i+2]))
    """

def generate_minhash(tokens: list):
    m = MinHash() # num_perm=256??
    for token in tokens:
        m.update(token.encode('utf-8'))
    return m

def generate_minhash_lsh(threshold: float):
    return MinHashLSH(threshold=threshold) # num_perm=256??

def duplicates_prevention(code: str, lsh: MinHashLSH) -> Tuple[bool, MinHash]:
    minhash = generate_minhash(tokenize_code(code))
    return len(lsh.query(minhash)) > 0, minhash

def insert_minhash_to_lsh(lsh: MinHashLSH, mh: MinHash):
    unique_key = uuid.uuid4().hex 
    lsh.insert(key=unique_key, minhash=mh)
    return lsh