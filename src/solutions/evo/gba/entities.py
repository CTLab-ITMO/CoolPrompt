from typing import Type, List, Tuple
import numpy as np
from scipy.special import softmax
from src.solutions.evo.base import Prompt, PromptOrigin


class Player(Prompt):

    def __init__(
        self,
        name: str,
        text: str,
        origin: PromptOrigin = PromptOrigin.EVOLUTED,
        score: float = None,
        retire_timer: int = 4,
    ) -> None:
        self.name = name
        self.retire_timer = retire_timer
        self.heuristic = ""

        super().__init__(
            text=text,
            origin=origin,
            score=score
        )

    def add_year(self) -> None:
        self.retire_timer -= 1

    def set_heuristic(self, heuristic: str) -> None:
        self.heuristic = heuristic

    def retired(self) -> bool:
        return self.retire_timer <= 0

    @classmethod
    def from_prompt(
        cls: Type['Player'],
        prompt: Prompt,
        name: str
    ) -> 'Player':
        return cls(
            name=name,
            text=prompt.text,
            origin=prompt.origin,
            score=prompt.score,
        )

    def __str__(self) -> str:
        return f"""
            {self.name}
            Text: {self.text}
            Score: {self.score}
        """

    def to_dict(self) -> dict:
        data = super().to_dict()
        data['name'] = str(self.name)
        data['heuristic'] = str(self.heuristic)
        return data


class Manager:

    def __init__(
        self,
        name: str,
        style: str,
        long_term_reflection: str,
        group_training_template: str,
        individual_training_template: str,
        heuristic: str = 'lhh',
    ) -> None:
        self.name = name
        self.style = style
        self.long_term_reflection = long_term_reflection
        self.successful_training = False
        self.group_training_template = group_training_template
        self.individual_training_template = individual_training_template
        self.heuristic = heuristic

    def update_reflection(self, reflection: str) -> None:
        self.long_term_reflection = reflection

    def group_training_request(self, **args):
        if self.heuristic == 'lhh':
            return self.group_training_template.replace(
                    "<STYLE>",
                    self.style
                ).replace(
                    "<PROBLEM_DESCRIPTION>",
                    args['problem_description']
                ).replace(
                    "<EXAMPLES>",
                    args['examples']
                ).replace(
                    "<REFLECTION>",
                    args['reflection']
                )
        if self.heuristic == 'ga':
            return self.group_training_template.replace(
                "<PROMPTS>",
                args['prompts']
            )
        return self.group_training_template.replace(
            "<PROMPT1>",
            args['prompt1'],
        ).replace(
            "<PROMPT2>",
            args['prompt2'],
        ).replace(
            "<PROMPT3>",
            args['prompt3']
        ).replace(
            "<ELITIST>",
            args['elitist']
        )

    def individual_training_request(self, **args):
        if self.heuristic == 'llh':
            return self.individual_training_template.replace(
                    "<STYLE>",
                    self.style
                ).replace(
                    "<PROBLEM_DESCRIPTION>",
                    args['problem_description']
                ).replace(
                    "<PROMPT>",
                    args['prompt']
                ).replace(
                    "<REFLECTION>",
                    self.long_term_reflection
                )
        return self.individual_training_template.replace(
            "<PROMPT>",
            args['prompt']
        )

    def __str__(self) -> str:
        return f"""
            {self.name}
            Style: {self.style}
            Reflection: {self.long_term_reflection}
            Heuristic: {self.heuristic}
        """

    def to_dict(self) -> dict:
        data = {}
        data['name'] = str(self.name)
        data['style'] = str(self.style)
        data['reflection'] = str(self.long_term_reflection)
        data['heuristic'] = str(self.heuristic)
        return data


class Team:

    def __init__(self, name: str) -> None:
        self.name = name
        self.manager = None
        self.players = []
        self.bad_training_seasons = 0

    def sign_player(self, player: Player) -> None:
        self.players.append(player)

    def sign_manager(self, manager: Manager) -> None:
        self.manager = manager
        self.bad_training_seasons = 0

    def power(self) -> float:
        return float(np.mean([player.score for player in self.players]))

    def __str__(self) -> str:
        players_str = '\n'.join([str(player) for player in self.players])
        return f"""{self.name}
            Power: {self.power()}
            Manager: {str(self.manager)}
            Players: {players_str}
        """

    def to_dict(self) -> dict:
        data = {}
        data['name'] = str(self.name)
        data['manager'] = self.manager.to_dict()
        data['players'] = [player.to_dict() for player in self.players]
        data['rating'] = float(self.power())
        return data


class League:

    def __init__(self, teams: List[Team]) -> None:
        self.teams = teams
        self.table = {}
        self.tour = 0

    def _next_cycle(self, indices: List[int]) -> None:
        tmp = indices[1]
        for i in range(2, len(indices)):
            indices[i - 1] = indices[i]
        indices[-1] = tmp

    def _make_tour(self, indices: List[int]) -> List[Tuple[int, int]]:
        tour = []
        i = 0
        j = len(indices) - 1
        while i < j:
            tour.append((indices[i], indices[j]))
            i += 1
            j -= 1
        return tour

    def _make_schedule(self) -> None:
        indices = np.arange(len(self.teams))
        np.random.shuffle(indices)
        self.schedule = []
        for _ in range(len(self.teams) - 1):
            self.schedule.append(self._make_tour(indices))
            self._next_cycle(indices)

    def start_season(self) -> None:
        self._make_schedule()
        self.table = {}
        for i in range(len(self.teams)):
            self.table[i] = 0
            self.teams[i].manager.successful_training = False

    def finished(self) -> bool:
        return self.tour >= len(self.schedule)

    def _game(self, team1: Team, team2: Team) -> Tuple[int, int]:
        power1 = team1.power()
        power2 = team2.power()
        p = softmax([power1 * 100, power2 * 100])
        k = 1 / (1 + abs(p[0] - p[1]))
        probas = np.array([p[0], k, p[1]])
        probas /= sum(probas)
        ind = np.random.choice(range(3), p=probas)
        if ind == 0:
            return 3, 0
        if ind == 1:
            return 1, 1
        return 0, 3

    def play_tour(self) -> List[str]:
        games = self.schedule[self.tour]
        output = []
        for ind1, ind2 in games:
            team1 = self.teams[ind1]
            team2 = self.teams[ind2]
            points = self._game(team1, team2)
            output.append(
                f"{team1.name} - {team2.name}      {points[0]} : {points[1]}"
            )
            self.table[ind1] += points[0]
            self.table[ind2] += points[1]
        self.tour += 1
        return output

    def _get_ranked_indices(self) -> List[int]:
        indices = range(len(self.teams))
        indices = list(
            sorted(indices, key=lambda i: self.table[i], reverse=True)
        )
        return indices

    def end_of_season(self) -> List[Team]:
        indices = self._get_ranked_indices()
        new_order = [self.teams[i] for i in indices]
        self.teams = new_order
        self.tour = 0
        self.schedule = []
        return self.teams

    def get_dict_table(self) -> dict:
        data = {}
        data['tour'] = self.tour
        indices = self._get_ranked_indices()
        data['table'] = [
            {
                'team': str(self.teams[i].name),
                'points': self.table[i]
            }
            for i in indices
        ]
        return data
