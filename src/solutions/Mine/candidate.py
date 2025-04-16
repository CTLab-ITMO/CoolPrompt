


class Candidate:
    def __init__(self, prompt: str, train_score: float):
        self.prompt = prompt
        self.train_score = train_score
        
        
class CandidateHistory:
    def __init__(self, candidates: list[Candidate] | None = None):
        self.candidates = []
        if candidates:
            self.candidates.extend(candidates)
        
        
    def add(self, cand: Candidate) -> None:
        self.candidates.append(cand)
        
    def extend(self, candidates: list[Candidate]) -> None:
        self.candidates.extend(candidates)
        
    def clear(self) -> None:
        self.candidates = []
    
    # strategy 1
    def get_highest_scorer(self) -> Candidate:
        "return candidate with maximum train_score"
        if not self.candidates:
            raise Exception("No candidates in history")
        return max(self.candidates, key=lambda cand: cand.train_score)
    
    # strategy 2:
    # pick based on length as well, like a punishing factor
    
    # strategy 3: dynamic one, first few - best score, then - length.