from concurrent.futures import ThreadPoolExecutor, as_completed

from src.preference_generation.candidate_generator import CandidateGenerator
from src.preference_generation.docker_sandbox import DockerSandbox
from src.data_curation.validators import linter_check
from src.preference_generation.preference_generation_config import W_EXEC, W_COMPLEXITY, W_LINT

class PreferenceOrchestrator:
    def __init__(self, w_exec=W_EXEC, w_complexity=W_COMPLEXITY, w_lint=W_LINT):
        self.candidate_generator = CandidateGenerator()
        self.sandbox = DockerSandbox()
        self.w_exec = w_exec
        self.w_complexity = w_complexity
        self.w_lint = w_lint

    def reward_score(self, code: str, language: str) -> dict:
        try: 
            execution = self.sandbox.execute_code(code, language)
            passed = execution["success"]
            r_exec = 1.0 if passed else 0.0
            
            metrics = linter_check(code, language)
            lint_errors = metrics.get("lint_errors")
            complexity = metrics.get("complexity")
            
            score = (self.w_exec * r_exec)
            
            if complexity is not None:
                score -= (self.w_complexity * float(complexity))
            if lint_errors is not None:
                score -= (self.w_lint * float(lint_errors))
            
            return {
                "score": score,
                "passed": passed,
                "lint_errors": lint_errors,
                "complexity": complexity,
                "stdout": execution.get("stdout", ""),
                "stderr": execution.get("stderr", ""),
            }
        except FileNotFoundError:
            raise
        except Exception as e:
            return {
                "score": -999.0,
                "passed": False,
                "lint_errors": None,
                "complexity": None,
                "stdout": "",
                "stderr": str(e),
            }

    def _build_result(self, prompt, language, candidates, evals, chosen_idx, rejected_idx, case):
        chosen_eval = evals[chosen_idx]
        rejected_eval = evals[rejected_idx]
        return {
            "prompt": prompt,
            "chosen": candidates[chosen_idx],
            "rejected": candidates[rejected_idx],
            "case": case,
            "language": language,
            "discarded": False,
            "chosen_score": chosen_eval["score"],
            "rejected_score": rejected_eval["score"],
            "chosen_complexity": chosen_eval["complexity"],
            "rejected_complexity": rejected_eval["complexity"],
            "chosen_lint_errors": chosen_eval["lint_errors"],
            "rejected_lint_errors": rejected_eval["lint_errors"],
        }

    def _build_discard(self, prompt, language, candidates, evals, case):
        result = {
            "prompt": prompt, "case": case, "language": language, "discarded": True,
        }
        for j, (cand, ev) in enumerate(zip(candidates, evals)):
            result[f"candidate_{j}_code"] = cand[:500]
            result[f"candidate_{j}_stderr"] = ev.get("stderr", "")[:300]
            result[f"candidate_{j}_score"] = ev.get("score")
        return result

    def _classify_pair(self, prompt: str, language: str, candidates: list[str], evals: list[dict]) -> dict:
        evaluations = {"passing": [], "failing": []}
        for j, ev in enumerate(evals):
            if ev["passed"]:
                evaluations["passing"].append((j, ev["score"]))
            else:
                evaluations["failing"].append((j, ev["score"]))

        if not evaluations["passing"]:
            scores = [(j, ev["score"]) for j, ev in enumerate(evals)]
            best = max(scores, key=lambda x: x[1])
            worst = min(scores, key=lambda x: x[1])

            if best[0] == worst[0] or abs(best[1] - worst[1]) < 1e-6:
                return self._build_discard(prompt, language, candidates, evals, "B_tied")

            return self._build_result(prompt, language, candidates, evals, chosen_idx=best[0], rejected_idx=worst[0], case="B")

        if evaluations["passing"] and evaluations["failing"]:
            chosen_idx = max(evaluations["passing"], key=lambda x: x[1])[0]
            rejected_idx = min(evaluations["failing"], key=lambda x: x[1])[0]
            return self._build_result(prompt, language, candidates, evals,
                                      chosen_idx, rejected_idx, case="A")

        best = max(evaluations["passing"], key=lambda x: x[1])
        worst = min(evaluations["passing"], key=lambda x: x[1])
        chosen_idx, rejected_idx = best[0], worst[0]

        if chosen_idx == rejected_idx or abs(best[1] - worst[1]) < 1e-6:
            return self._build_discard(prompt, language, candidates, evals, "C_tied")

        return self._build_result(prompt, language, candidates, evals, chosen_idx, rejected_idx, case="C")

    def create_preference_pair(self, prompt: str, language: str) -> dict:
        candidates = self.candidate_generator.generate_candidates(prompt)
        evals = [self.reward_score(c, language) for c in candidates]
        return self._classify_pair(prompt, language, candidates, evals)

    def create_preference_pairs_batch(self, prompts: list[str], languages: list[str]) -> list[dict]:
        all_candidates = self.candidate_generator.generate_candidates_batch(prompts)

        eval_results: dict[tuple[int, int], dict] = {}
        max_workers = min(len(prompts) * 2, 8)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for i, (candidates, lang) in enumerate(zip(all_candidates, languages)):
                for j, candidate in enumerate(candidates):
                    future = executor.submit(self.reward_score, candidate, lang)
                    futures[future] = (i, j)

            for future in as_completed(futures):
                key = futures[future]
                eval_results[key] = future.result()

        results = []
        for i, (prompt, lang) in enumerate(zip(prompts, languages)):
            candidates = all_candidates[i]
            evals = [eval_results[(i, j)] for j in range(len(candidates))]
            results.append(self._classify_pair(prompt, lang, candidates, evals))

        return results