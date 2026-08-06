from src.preference_generation.candidate_generator import CandidateGenerator
from src.preference_generation.docker_sandbox import DockerSandbox
from src.data_curation.validators import linter_check
from src.preference_generation.preference_generation_config import W_EXEC, W_COMPLEXITY, W_LINT

class PreferenceOrchestrator:
    def __init__(self):
        self.candidate_generator = CandidateGenerator()
        self.sandbox = DockerSandbox()

    def reward_score(self, code: str, language: str) -> dict:
        try: 
            execution = self.sandbox.execute_code(code, language)
            passed = execution["success"]
            r_exec = 1.0 if passed else 0.0
            
            metrics = linter_check(code, language)
            lint_errors = metrics.get("lint_errors")
            complexity = metrics.get("complexity")
            
            score = (W_EXEC * r_exec)
            
            if complexity is not None:
                score -= (W_COMPLEXITY * float(complexity))
            if lint_errors is not None:
                score -= (W_LINT * float(lint_errors))
            
            return {
                "score": score,
                "passed": passed,
                "lint_errors": lint_errors,
                "complexity": complexity,
                "stdout": execution.get("stdout", ""),
                "stderr": execution.get("stderr", ""),
            }
        except Exception as e:
            return {
                "score": -999.0,
                "passed": False,
                "lint_errors": None,
                "complexity": None,
                "stdout": "",
                "stderr": str(e),
            }

    def create_preference_pair(self, prompt: str, language: str) -> dict:
        """
        Case classification:
            A — one passes, one fails → chosen = passing
            B — both fail → discarded (noisy gradients)
            C — both pass → ranked by composite score
            C_tied — both pass with identical scores → discarded
        """
        candidates = self.candidate_generator.generate_candidates(prompt)

        eval_results = []
        evaluations = {
            "passing": [],
            "failing": [],
        }
        
        for idx, candidate in enumerate(candidates):
            eval_result = self.reward_score(candidate, language)
            eval_results.append(eval_result)
            if eval_result["passed"]:
                evaluations["passing"].append((idx, eval_result["score"]))
            else:
                evaluations["failing"].append((idx, eval_result["score"]))

        # Case B — all fail → discard (prevents noisy gradients)
        if not evaluations["passing"]:
            return {
                "prompt": prompt,
                "case": "B",
                "language": language,
                "discarded": True,
            }

        # Case A — at least one passes and at least one fails
        if evaluations["passing"] and evaluations["failing"]:
            case = "A"
            chosen_idx = max(evaluations["passing"], key=lambda x: x[1])[0]
            rejected_idx = min(evaluations["failing"], key=lambda x: x[1])[0]

        # Case C — all pass → rank by composite quality metric
        else:
            chosen_idx = max(evaluations["passing"], key=lambda x: x[1])[0]
            rejected_idx = min(evaluations["passing"], key=lambda x: x[1])[0]
            
            if chosen_idx == rejected_idx:
                return {
                    "prompt": prompt,
                    "case": "C_tied",
                    "language": language,
                    "discarded": True,
                }
            case = "C"

        chosen_eval = eval_results[chosen_idx]
        rejected_eval = eval_results[rejected_idx]

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