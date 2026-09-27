set -euo pipefail

HARNESS_REPO="https://github.com/bigcode-project/bigcode-evaluation-harness.git"
HARNESS_SHA="8fc5bae6479c4fbbb28c3f8b644f6a15b3f3b5bd"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIR="$ROOT/tools/bigcode-evaluation-harness"

if [ ! -d "$DIR/.git" ]; then
  git clone --quiet "$HARNESS_REPO" "$DIR"
fi
git -C "$DIR" cat-file -e "${HARNESS_SHA}^{commit}" 2>/dev/null || git -C "$DIR" fetch --quiet origin
git -C "$DIR" checkout --quiet --force --detach "$HARNESS_SHA"  
git -C "$DIR" clean -fdq                                     
for patch in "$ROOT"/patches/bigcode_harness_*.patch; do
  git -C "$DIR" apply --check "$patch"
  git -C "$DIR" apply "$patch"
  echo "applied $(basename "$patch")"
done

echo "bigcode-evaluation-harness @ ${HARNESS_SHA:0:7} (+ patches) ready in tools/"
echo "Evaluate every model with the same flags: bash scripts/05_run_eval_harness.sh python"
