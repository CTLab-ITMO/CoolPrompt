import os
from random import randint
from typing import List, Tuple
import numpy as np
from scipy.special import softmax
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from src.evaluation.evaluator import BaseNLPEvaluator
from src.solutions.evo.base import Evoluter, Prompt, PromptOrigin
from src.solutions.evo.self_evo.utils import parse_output, append_to_yaml
from src.solutions.evo.gba.entities import Team, Player, Manager, League


class GBAEvoluter(Evoluter):

    def __init__(
        self,
        model_name: str,
        dataset: str,
        evaluator: BaseNLPEvaluator,
        metric: str,
        task: str,
        teams: int = 5,
        players_per_team: int = 2,
        num_seasons: int = 10,
        output_path: str = './outputs',
        use_cache: bool = True,
        batch_size: int = 64,
    ) -> None:
        model = LLM(
            model=model_name,
            dtype=torch.float16,
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side='left'
        )

        super().__init__(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            evaluator=evaluator,
            metric=metric,
            task=task,
            use_cache=use_cache,
            batch_size=batch_size
        )

        self.config_filename = 'config.yaml'
        self.problem_description_filename = 'problems.yaml'
        self._mutation_prompts_filename = 'mutation_prompts.yaml'
        self._styles_of_thinking_filename = 'styles.yaml'
        self.output_path = output_path

        self.elitist = None
        self.best_score_overall = None
        self.best_prompt_overall = None
        self._config_path = './data'

        self.problem_description = self._read_yaml_data(
            os.path.join(
                self._config_path,
                self.problem_description_filename
            ),
            key=self.dataset
        )
        self._style_of_thinkings = self._read_yaml_data(
            os.path.join(
                self._config_path,
                self._styles_of_thinking_filename
            ),
            key='styles'
        )
        self._group_training_reflection_template = self._read_yaml_data(
            os.path.join(
                self._config_path,
                self.config_filename
            ),
            key='group_training_reflection'
        )
        self._group_training_template = self._read_yaml_data(
            os.path.join(
                self._config_path,
                self.config_filename
            ),
            key='group_training'
        )
        self._mutation_prompts = self._read_yaml_data(
            os.path.join(
                self._config_path,
                self._mutation_prompts_filename
            ),
            key='mutation_prompts'
        )
        self._long_term_reflection_template = self._read_yaml_data(
            os.path.join(
                self._config_path,
                self.config_filename
            ),
            key='long_term_reflection'
        )
        self._individual_training_template = self._read_yaml_data(
            os.path.join(
                self._config_path,
                self.config_filename
            ),
            key='individual_training'
        )

        self._player_names = self._read_names('player_names.yaml')
        self._manager_names = self._read_names('manager_names.yaml')
        self._team_names = self._read_names('team_names.yaml')
        self.teams_num = teams
        assert self.teams_num % 2 == 0
        self.players_per_team = players_per_team
        self.num_seasons = num_seasons

    def _managers_init(self, size: int = None) -> List[Manager]:
        if size is None:
            size = self.teams_num
        managers = []
        for _ in range(size):
            ind1 = randint(0, len(self._style_of_thinkings) - 1)
            ind2 = randint(0, len(self._mutation_prompts) - 1)
            manager = Manager(
                name=self._name(mode='manager'),
                style=self._style_of_thinkings[ind1],
                long_term_reflection=self._mutation_prompts[ind2]
            )
            managers.append(manager)
        return managers

    def _create_teams(self) -> List[Team]:
        return [Team(self._name(mode='team')) for _ in range(self.teams_num)]

    def _name(
        self,
        mode: bool = 'player',
        free: bool = False,
        name_to_free: str = None
    ) -> str:
        if mode == 'player':
            dict_names = self._player_names
        elif mode == 'manager':
            dict_names = self._manager_names
        else:
            dict_names = self._team_names
        available = [name for name, used in dict_names.items() if used is free]
        if free:
            chosen = name_to_free
        else:
            chosen = np.random.choice(np.array(available))
        if mode == 'player':
            self._player_names[chosen] = not free
        elif mode == 'manager':
            self._manager_names[chosen] = not free
        else:
            self._team_names[chosen] = not free
        return chosen

    def _read_names(self, filename: str) -> dict:
        names = self._read_yaml_data(
            os.path.join(
                self._config_path,
                filename
            ),
            key='names'
        )
        return {name: False for name in names}

    def _init_teams(self) -> List[Team]:
        cached_data = self._read_yaml_data(
            os.path.join(
                self._prompts_directory_path,
                'cached_prompts.yaml'
            )
        )
        if not cached_data:
            manual = self._read_prompts(
                'prompts.yaml',
                origin=PromptOrigin.MANUAL
            )
            ape = self._read_prompts(
                'prompts_auto.yaml',
                origin=PromptOrigin.APE
            )
            prompts = manual + ape
            self._evaluation(prompts)
            self._cache_population(
                prompts,
                os.path.join(
                    self._prompts_directory_path,
                    'cached_prompts.yaml'
                )
            )
        else:
            prompts = [
                Prompt.from_dict(prompt_data) for prompt_data in cached_data
            ]

        players = np.array([
            Player.from_prompt(prompt, name=self._name())
            for prompt in prompts
        ])
        players = self._reranking(players)
        players = players[:self.teams_num * self.players_per_team]
        np.random.shuffle(players)
        teams = self._create_teams()
        for i in range(self.teams_num):
            team = teams[i]
            shift = i * self.players_per_team
            for j in range(self.players_per_team):
                player = players[shift + j]
                team.sign_player(player)
        managers = self._managers_init()
        for team, manager in zip(teams, managers):
            team.sign_manager(manager)
        return teams

    def _llm_query(
        self,
        requests: List[str],
        verbose: bool = False,
        **config
    ) -> List[str]:
        """Provides api to query requests to the model.

        Args:
            requests (List[str]): string requests.
            verbose (bool, optional): Whether to use logging or not.
                Defaults to False.
            config: additional sampling params.

        Returns:
            List[str]: model answers.
        """
        sampling_params = {
            "max_tokens": 150,
            "top_p": 0.95,
            "frequency_penalty": 0,
            "presence_penalty": 0,
        }
        sampling_params.update(**config)
        sampling_params = SamplingParams(**sampling_params)

        answers = self.model.generate(
            prompts=requests,
            sampling_params=sampling_params,
            use_tqdm=verbose
        )

        results = [answer.outputs[0].text for answer in answers]
        return results

    def _group_training_reflection(
        self,
        teams: List[Team],
        season: int,
        tour: int,
        verbose: bool = True
    ) -> Tuple[List[str], List[str]]:
        reflection_requests = []
        team_examples = []
        for team in teams:
            players = team.players
            players = np.array(
                sorted(players, key=lambda p: p.score, reverse=True)
            )
            examples = '\n'.join([p.text for p in players])
            team_examples.append(examples)
            request = self._group_training_reflection_template.replace(
                "<STYLE>",
                team.manager.style
            ).replace(
                "<PROBLEM_DESCRIPTION>",
                self.problem_description
            ).replace(
                "<EXAMPLES>",
                examples
            )
            reflection_requests.append(request)
        reflections = self._llm_query(
            reflection_requests,
            verbose=verbose,
            temperature=0.1
        )
        self._cache_data(
            [
                {
                    str(team.name): str(reflection)
                }
                for team, reflection in zip(teams, reflections)
            ],
            os.path.join(
                self.output_path,
                f"Season {season}",
                f"Tour {tour}",
                "short_reflections.yaml",
            )
        )
        reflections = [
            parse_output(reflection, bracket='<hint>')
            for reflection in reflections
        ]
        return reflections, team_examples

    def _long_term_reflection(
        self,
        teams: List[Team],
        short_term_reflections: List[str],
        season: int,
        tour: int,
        verbose: bool = True
    ) -> None:
        requests = []
        for team, short_term_reflection in zip(teams, short_term_reflections):
            request = self._long_term_reflection_template.replace(
                "<STYLE>",
                team.manager.style
            ).replace(
                "<PROBLEM_DESCRIPTION>",
                self.problem_description
            ).replace(
                "<LONG_TERM_REFLECTION>",
                team.manager.long_term_reflection
            ).replace(
                "<SHORT_TERM_REFLECTION>",
                short_term_reflection
            )
            requests.append(request)
        responses = self._llm_query(requests, verbose=verbose)
        responses = [
            parse_output(response, bracket='<hint>') for response in responses
        ]
        cache_data = []
        for team, response in zip(teams, responses):
            team.manager.update_reflection(response)
            cache_data.append(
                {
                    str(team.manager.name):
                        {
                            'team': str(team.name), 'reflection': str(response)
                        }
                }
            )
        self._cache_data(
            cache_data,
            os.path.join(
                self.output_path,
                f"Season {season}",
                f"Tour {tour}",
                "long_reflections.yaml",
            )
        )

    def _mutate(self, population):
        return super()._mutate(population)

    def _selection(self, population):
        return super()._selection(population)

    def _update_player(
        self,
        team: Team,
        prompt: Prompt,
        index: int = None
    ) -> bool:
        players = team.players
        if index is None:
            upgradeable = [p for p in players if p.score < prompt.score]
            if len(upgradeable) == 0:
                self.logger.info(f"Nobody to upgrade in {team.name}")
                return False
            player_to_upgrade = np.random.choice(upgradeable)
        else:
            player_to_upgrade = players[index]
            if player_to_upgrade.score >= prompt.score:
                self.logger.info(
                    f"No need in training for {player_to_upgrade.name}"
                )
                return False
        player_to_upgrade.text = prompt.text
        player_to_upgrade.score = prompt.score
        player_to_upgrade.origin = prompt.origin
        self.logger.info(f"""
                         {'=' * 50}
                         {player_to_upgrade.name}
                         Text: {player_to_upgrade.text}
                         Score: {player_to_upgrade.score}
                         {'=' * 50}
                         """)
        return True

    def _group_training(
        self,
        teams: List[Team],
        season: int,
        tour: int,
        verbose: bool = True
    ) -> None:
        self.logger.info("Group training")
        reflections, examples = self._group_training_reflection(
            teams,
            season,
            tour,
            verbose
        )
        self._long_term_reflection(
            teams,
            reflections,
            season,
            tour,
            verbose=verbose
        )
        requests = []
        for team, reflection, example in zip(teams, reflections, examples):
            request = self._group_training_template.replace(
                "<STYLE>",
                team.manager.style
            ).replace(
                "<PROBLEM_DESCRIPTION>",
                self.problem_description
            ).replace(
                "<EXAMPLES>",
                example
            ).replace(
                "<REFLECTION>",
                reflection
            )
            requests.append(request)
        responses = self._llm_query(requests, verbose=verbose, temperature=0.1)
        responses = [parse_output(response) for response in responses]
        new_prompts = [
            Prompt(response) for response in responses
        ]
        for team, new_prompt in zip(teams, new_prompts):
            self._evaluate(new_prompt)
            self._update_player(team, new_prompt)

    def _individual_training(
        self,
        teams: List[Team],
        verbose: bool = True
    ) -> None:
        self.logger.info("Individual training")
        requests = []
        for team in teams:
            for player in team.players:
                request = self._individual_training_template.replace(
                    "<STYLE>",
                    team.manager.style
                ).replace(
                    "<PROBLEM_DESCRIPTION>",
                    self.problem_description
                ).replace(
                    "<PROMPT>",
                    player.text
                ).replace(
                    "<REFLECTION>",
                    team.manager.long_term_reflection
                )
                requests.append(request)
        responses = self._llm_query(requests, verbose=verbose, temperature=0.3)
        responses = [parse_output(response) for response in responses]
        for i, team in enumerate(teams):
            self.logger.info(team.name)
            for j in range(len(team.players)):
                new_prompt = Prompt(
                    responses[i * self.players_per_team + j],
                    origin=PromptOrigin.MUTATED
                )
                self._evaluate(new_prompt)
                self._update_player(team, new_prompt, index=j)

    def _update_elitist(self, teams: List[Team]) -> None:
        for team in teams:
            for player in team.players:
                if (
                    self.best_score_overall is None
                    or self.best_score_overall <= player.score
                ):
                    self.elitist = player
                    self.best_score_overall = player.score
                    self.best_prompt_overall = player.text

    def _transfers(self, teams: List[Team], season: int) -> None:
        for team in teams:
            team.players = self._reranking(team.players)
        free_agents = []
        transfers = []
        for i in range(len(teams) - 1):
            best_player = teams[i + 1].players[0]
            worst_player = teams[i].players[-1]
            if best_player.score >= worst_player.score:
                self.logger.info(
                    f"Transfer: {best_player.name}\t" +
                    f"{teams[i + 1].name} --> {teams[i].name}"
                )
                transfers.append(
                    {
                        'player': str(best_player.name),
                        'score': float(best_player.score),
                        'from': str(teams[i + 1].name),
                        'to': str(teams[i].name)
                    }
                )
                teams[i].players[-1] = best_player
                free_agents.append(worst_player)
                teams[i + 1].players = teams[i + 1].players[1:]

        teams_to_fill = [
            team for team in teams if len(team.players) < self.players_per_team
        ]
        free_agents = self._reranking(free_agents)
        while len(teams_to_fill) > 0:
            free_player = free_agents[0]
            scores = [team.power() for team in teams_to_fill]
            probas = softmax(scores)
            team_ind = np.random.choice(range(len(teams_to_fill)), p=probas)
            team = teams_to_fill[team_ind]
            self.logger.info(f"Signing: {free_player.name} to {team.name}")
            transfers.append(
                {
                    'player': str(free_player.name),
                    'score': float(free_player.score),
                    'to': str(team.name)
                }
            )
            team.sign_player(free_player)
            teams_to_fill.pop(team_ind)
            free_agents.pop(0)
        self._cache_data(
            transfers,
            os.path.join(
                self.output_path,
                f"Season {season}",
                "transfers.yaml",
            )
        )

    def _change_managers(self, teams: List[Team]) -> None:
        start = len(teams) // 2
        new_managers = self._managers_init(size=len(teams) - start)
        for i, manager in zip(range(start, len(teams)), new_managers):
            old_manager = teams[i].manager
            teams[i].sign_manager(manager)
            self.logger.info(f"{teams[i].name} new manager: {manager.name}")
            self._name(
                mode='manager',
                free=True,
                name_to_free=old_manager.name
            )

    def evolution(self) -> None:
        teams = self._init_teams()
        self._cache_data(
            [{'team': team.to_dict()} for team in list(teams)],
            os.path.join(
                self.output_path,
                "initial_teams.yaml"
            )
        )

        league = League(teams)
        for season in range(self.num_seasons):
            league.start_season()
            self.logger.info(f"Season #{season} has started")

            while not league.finished():
                self.logger.info(f"Tour #{league.tour}")
                log_outputs = league.play_tour()
                table = league.get_dict_table()
                table['matches'] = log_outputs
                self._cache_data(
                    table,
                    os.path.join(
                        self.output_path,
                        f"Season {season}",
                        f"Tour {league.tour}",
                        "tour.yaml",
                    )
                )

                if randint(0, 1) == 0:
                    self._group_training(
                        teams,
                        season,
                        league.tour,
                        verbose=True
                    )
                else:
                    self._individual_training(teams, verbose=True)
                self._update_elitist(teams)
                self._cache_data(
                    [{'team': team.to_dict()} for team in list(teams)],
                    os.path.join(
                        self.output_path,
                        f"Season {season}",
                        f"Tour {league.tour}",
                        "teams.yaml",
                    )
                )

            teams = league.end_of_season()
            self.logger.info(f"Season #{season} has finished")
            self._cache_data(
                {
                    'best_player': self.elitist.to_dict(),
                    'best_team': str(teams[0].name),
                    'teams': [
                        {
                            'name': str(team.name),
                            'rating': float(team.power())
                        }
                        for team in teams
                    ]
                },
                os.path.join(
                    self.output_path,
                    f"Season {season}",
                    "results.yaml",
                )
            )
            self._transfers(teams, season)
            self._change_managers(teams)
        self.logger.info(f"BEST SCORE: {self.best_score_overall}")
        self.logger.info(f"BEST PROMPT:\n{self.best_prompt_overall}")

        all_players = np.concatenate([team.players for team in teams])
        all_players = self._reranking(all_players)
        all_players = all_players[:3]
        self._evaluation(all_players, split='test')
        self._cache_population(
            [p.to_dict() for p in all_players],
            self._make_output_path('best_prompts_infer.yaml')
        )
        append_to_yaml(
            new_data={
                self.dataset: all_players[0].to_dict(),
            },
            filepath="./best_prompts.yaml"
        )
