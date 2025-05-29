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
        version: str = 'v1'
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

        self.version = version
        self.config_filename = f'config_{self.version}.yaml'
        self.problem_description_filename = 'problems.yaml'
        self._mutation_prompts_filename = 'mutation_prompts_v2.yaml'
        self._styles_of_thinking_filename = 'styles_v2.yaml'
        self.output_path = output_path

        self.elitist = None
        self._long_term_reflection_str = ""
        self.best_score_overall = None
        self.best_prompt_overall = ""
        self._free_agents = []
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
        self._global_long_term_reflection_template = self._read_yaml_data(
            os.path.join(
                self._config_path,
                self.config_filename
            ),
            key='global_long_term_reflection'
        )
        self._academy_mutation_template = self._read_yaml_data(
            os.path.join(
                self._config_path,
                self.config_filename
            ),
            key='academy_mutation'
        )
        self._crossover_ga_template = self._read_yaml_data(
            os.path.join(
                self._config_path,
                self.config_filename
            ),
            key='crossover_ga'
        )
        self._crossover_de_template = self._read_yaml_data(
            os.path.join(
                self._config_path,
                self.config_filename
            ),
            key='crossover_de'
        )
        self._paraphrase_template = self._read_yaml_data(
            os.path.join(
                self._config_path,
                self.config_filename
            ),
            key='paraphrase'
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
            heuristic = 'llh'
            gtt = self._group_training_template
            itt = self._individual_training_template
            h = randint(1, 10)
            if self.version == 'v4':
                if 6 < h <= 8:
                    heuristic = "ga"
                    gtt = self._crossover_ga_template
                    itt = self._paraphrase_template
                elif h >= 9:
                    heuristic = "de"
                    gtt = self._crossover_de_template
                    itt = self._paraphrase_template
            elif self.version == 'v5':
                if 8 <= h <= 9:
                    heuristic = "de"
                    gtt = self._crossover_de_template
                    itt = self._paraphrase_template
                elif h == 10:
                    heuristic = "ga"
                    gtt = self._crossover_ga_template
                    itt = self._paraphrase_template
            manager = Manager(
                name=self._name(mode='manager'),
                style=self._style_of_thinkings[ind1],
                long_term_reflection=self._mutation_prompts[ind2],
                group_training_template=gtt,
                individual_training_template=itt,
                heuristic=heuristic,
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
            "max_tokens": 300,
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
        reflections = [
            parse_output(reflection, bracket='<hint>')
            for reflection in reflections
        ]
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
        responses = self._llm_query(
            requests,
            verbose=verbose,
            temperature=0.15
        )
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
        cache_data.append({
            'global_reflection': self._long_term_reflection_str
        })
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
                self._free_agents.append(prompt)
                return False
            player_to_upgrade = np.random.choice(upgradeable)
        else:
            player_to_upgrade = players[index]
            if player_to_upgrade.score >= prompt.score:
                self.logger.info(
                    f"No need in training for {player_to_upgrade.name}"
                )
                self._free_agents.append(prompt)
                return False
        team.manager.successful_training = True
        player_to_upgrade.text = prompt.text
        player_to_upgrade.score = prompt.score
        player_to_upgrade.origin = prompt.origin
        player_to_upgrade.set_heuristic(team.manager.heuristic)
        self.logger.info(f"""
                         {'=' * 50}
                         {player_to_upgrade.name}
                         Text: {player_to_upgrade.text}
                         Score: {player_to_upgrade.score}
                         {'=' * 50}
                         """)
        return True

    def _global_long_term_reflection(
        self,
        short_term_reflections: List[str],
        verbose: bool = False
    ) -> None:
        """Long-term reflection before mutation.

        Args:
            short_term_reflections (List[str]): short-term reflections.
            verbose (bool, optional): Whether to use logging or not.
                Defaults to False.
        """
        request = self._global_long_term_reflection_template.replace(
            '<PROBLEM_DESCRIPTION>',
            self.problem_description
        ).replace(
            '<PRIOR_LONG_TERM_REFLECTION>',
            self._long_term_reflection_str
        ).replace(
            '<NEW_SHORT_TERM_REFLECTIONS>',
            '\n'.join(short_term_reflections)
        )

        response = self._llm_query(
            [request],
            verbose=verbose,
            temperature=0.15
        )[0]

        self._long_term_reflection_str = parse_output(
            response,
            bracket='<hint>'
        )

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
        self._global_long_term_reflection(reflections, verbose=verbose)
        self._long_term_reflection(
            teams,
            reflections,
            season,
            tour,
            verbose=verbose
        )
        requests = []
        for team, reflection, example in zip(teams, reflections, examples):
            request = team.manager.group_training_request(
                problem_description=self.problem_description,
                examples=example,
                reflection=reflection,
                prompts='\n'.join([
                    f"Prompt{i}: {p.text}" for i, p in enumerate(team.players)
                ]),
                prompt1=team.players[1].text,
                prompt2=team.players[2].text,
                prompt3=team.players[0].text,
                elitist=self.best_prompt_overall
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
                request = team.manager.individual_training_request(
                    problem_description=self.problem_description,
                    prompt=player.text,
                )
                requests.append(request)
        responses = self._llm_query(
            requests,
            verbose=verbose,
            temperature=0.15
        )
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
                self._free_agents.append(worst_player)
                teams[i + 1].players = teams[i + 1].players[1:]

        teams_to_fill = [
            team for team in teams if len(team.players) < self.players_per_team
        ]
        self._free_agents = self._reranking(self._free_agents)
        while len(teams_to_fill) > 0:
            free_player = self._free_agents[0]
            if isinstance(free_player, Prompt):
                free_player = Player.from_prompt(free_player, self._name())
                self.logger.info(
                    "Using training powers to create " +
                    f"{free_player.name} - {free_player.score}"
                )
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
            self._free_agents.pop(0)
        for p in self._free_agents:
            if isinstance(p, Player):
                self.logger.info(f"{p.name} is retiring")
                self._name(mode='player', free=True, name_to_free=p.name)
        self._free_agents = []
        self._cache_data(
            transfers,
            os.path.join(
                self.output_path,
                f"Season {season}",
                "transfers.yaml",
            )
        )

    def _change_manager(self, team: Team, manager: Manager) -> None:
        old_manager = team.manager
        team.sign_manager(manager)
        self.logger.info(
            f"{team.name} new manager: {manager.name}" +
            f" - {manager.heuristic}"
        )
        self._name(
            mode='manager',
            free=True,
            name_to_free=old_manager.name
        )

    def _change_managers(self, teams: List[Team]) -> None:
        start = len(teams) // 2
        new_managers = self._managers_init(size=len(teams) - start)
        for i, manager in zip(range(start, len(teams)), new_managers):
            self._change_manager(teams[i], manager)
        for i in range(start):
            if teams[i].manager.successful_training is False:
                teams[i].bad_training_seasons += 1
                if self.version == 'v4':
                    if teams[i].bad_training_seasons >= 2:
                        new_manager = self._managers_init(size=1)[0]
                        self.logger.info(
                            "After 2 seasons without any good training " +
                            f"{teams[i].name} fired {teams[i].manager.name}"
                        )
                        self._change_manager(teams[i], new_manager)
                elif self.version == 'v5':
                    if teams[i].bad_training_seasons >= 3:
                        new_manager = self._managers_init(size=1)[0]
                        self.logger.info(
                            "After 3 seasons without any good training " +
                            f"{teams[i].name} fired {teams[i].manager.name}"
                        )
                        self._change_manager(teams[i], new_manager)
            else:
                teams[i].bad_training_seasons = 0

    def _young_players_academy(
        self,
        teams: List[Team],
        season: int,
        verbose: bool = True
    ) -> None:
        elitists = self._captains(teams)
        request = self._academy_mutation_template.replace(
            '<PROBLEM_DESCRIPTION>',
            self.problem_description
        ).replace(
            '<LONG_TERM_REFLECTION>',
            self._long_term_reflection_str
        ).replace(
            '<ELITIST_PROMPTS>',
            '\n'.join([p.text for p in elitists])
        )
        responses = self._llm_query(
            [request] * (self.teams_num // 2),
            verbose=verbose,
            temperature=0.3
        )
        responses = [parse_output(response) for response in responses]
        new_prompts = [
            Prompt(response, origin=PromptOrigin.EVOLUTED)
            for response in responses
        ]
        start = self.teams_num // 2 + self.teams_num % 2
        academy_players = []
        for i, prompt in zip(range(start, self.teams_num), new_prompts):
            self._evaluate(prompt)
            players = teams[i].players
            players = self._reranking(players)
            if prompt.score >= players[-1].score:
                new_player = Player.from_prompt(prompt, self._name())
                old_player = players[-1]
                self.logger.info(
                    f"{teams[i].name} new academy player: " +
                    f"{new_player.name}, {new_player.score}\n" +
                    f"Instead of {old_player.name}, {old_player.score}"
                )
                players[-1] = new_player
                self._name(
                    mode='player',
                    free=True,
                    name_to_free=old_player.name
                )
                teams[i].players = players
                academy_players.append({
                    'player': new_player.to_dict(),
                    'old_player': old_player.to_dict(),
                    'team': str(teams[i].name)
                })
        self._cache_data(
            academy_players,
            os.path.join(
                self.output_path,
                f"Season {season}",
                "academy.yaml",
            )
        )

    def _elitists(self, teams: List[Team], n: int) -> List[Player]:
        all_players = np.concatenate([team.players for team in teams])
        all_players = self._reranking(all_players)
        all_players = all_players[:n]
        return all_players

    def _captains(self, teams: List[Team]) -> List[Player]:
        captains = []
        for team in teams:
            team.players = self._reranking(team.players)
            captains.append(team.players[0])
        return self._reranking(captains)

    def evolution(self) -> None:
        torch.cuda.empty_cache()
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
            self._young_players_academy(teams, season, verbose=True)
            self._change_managers(teams)
        self.logger.info(f"BEST SCORE: {self.best_score_overall}")
        self.logger.info(f"BEST PROMPT:\n{self.best_prompt_overall}")

        elitists = self._elitists(teams, n=3)
        self._evaluation(elitists, split='test')
        elitists = self._reranking(elitists)
        self._cache_data(
            [p.to_dict() for p in elitists],
            os.path.join(
                self.output_path,
                f"Season {season}",
                "best_prompts_infer.yaml",
            )
        )
        append_to_yaml(
            new_data={
                self.dataset: elitists[0].to_dict(),
            },
            filepath=f"./best_prompts_{self.version}.yaml"
        )
