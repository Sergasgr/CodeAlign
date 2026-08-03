from src.preference_generation.candidate_generator import CandidateGenerator
from src.preference_generation.docker_sandbox import DockerSandbox
from src.data_curation.validators import linter_check
from src.preference_generation.preference_generation_config import W_EXEC, W_COMPLEXITY, W_LINT

# R_total = (W_exec * R_exec) - (W_static * R_static) - (W_style * R_style)

class PreferenceOrchestrator:
    def __init__(self):
        self.candidate_generator = CandidateGenerator()
        self.sandbox = DockerSandbox()

    def reward_score(self, code: str, language: str) -> dict:
        # R_total = (W_exec * R_exec) - (W_complexity * complexity) - (W_lint * lint_errors)
        try: 
            execution = self.sandbox.execute_code(code, language)
            passed = execution["success"]
            r_exec = 1.0 if passed else 0.0
            
            metrics = linter_check(code, language)
            lint_errors = metrics.get("lint_errors") or 0
            complexity = metrics.get("complexity") or 0
            
            score = (W_EXEC * r_exec) - (W_COMPLEXITY * float(complexity)) - (W_LINT * float(lint_errors))
            
            return {
                "score": score,
                "passed": passed
            }
        except Exception:
            return {
                "score": -999.0,
                "passed": False
            }

    def create_preference_pair(self, prompt: str, language: str):
        candidates = self.candidate_generator.generate_candidates(prompt)
        evaluations = { 
            "passing": [],
            "failing": [],
        }
        
        for idx, candidate in enumerate(candidates):
            eval_result = self.reward_score(candidate, language)
            if eval_result["passed"]:
                evaluations["passing"].append((idx, eval_result["score"]))
            else:
                evaluations["failing"].append((idx, eval_result["score"]))

        if not evaluations["passing"]: # Discard the pair to prevent noisy gradients -> ALL FAILED
            return None
        if evaluations["passing"] and evaluations["failing"]:
            chosen_idx = max(evaluations["passing"], key=lambda x: x[1])[0]
            rejected_idx = min(evaluations["failing"], key=lambda x: x[1])[0]
        else:
            chosen_idx = max(evaluations["passing"], key=lambda x: x[1])[0]
            rejected_idx = min(evaluations["passing"], key=lambda x: x[1])[0]
            
            if chosen_idx == rejected_idx:
                return None
        
        return {
            "prompt": prompt,
            "chosen": candidates[chosen_idx],
            "rejected": candidates[rejected_idx]
        }