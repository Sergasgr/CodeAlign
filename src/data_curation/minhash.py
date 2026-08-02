"""Cross-sample near-duplicate detection (MinHash + LSH).
 
Distinct from code_smells.py: this catches redundant *training examples*
— two different samples that are near-identical to each other, possibly
copy-pasted across different repos in the dataset — not copy-pasted
blocks within one sample. It protects against wasting training signal /
overfitting on repeated examples across the corpus; code_smells.py is a
per-sample code-quality gate. Both stay — they catch different problems.
"""

import re
from datasketch import MinHash, MinHashLSH

TOKEN = re.compile(r"\w+|[^\w\s]")

def shingles(code: str, k: int = 5) -> set[str]:
    tokens = TOKEN.findall(code)
    if len(tokens) < k:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i:i + k]) for i in range(len(tokens) - k + 1)}

def signature(code: str, num_perm: int = 128) -> MinHash:
    mh = MinHash(num_perm=num_perm)
    for shingle in shingles(code):
        mh.update(shingle.encode("utf-8"))
    return mh

class DuplicateIndex:
    def __init__(self, threshold: float = 0.85, num_perm: int = 128):
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.num_perm = num_perm
        self.next_id = 0
 
    def check_and_add(self, code: str) -> str | None:
        mh = signature(code, self.num_perm)
        matches = self.lsh.query(mh)
        if matches:
            return str(matches[0])
        key = str(self.next_id)
        self.lsh.insert(key, mh)
        self.next_id += 1
        return None