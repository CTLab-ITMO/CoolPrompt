"""EvoPrompt evoluters adapted for the 8 generic datasets.

The class hierarchy mirrors the original BBH implementation
(:class:`Evoluter` → :class:`DEEvoluter`, :class:`GAEvoluter`,
:class:`ParaEvoluter`) but:

* loads data from ``datasets/data/<dataset>/{train,validation}_<dataset>.json``
* uses :func:`dataset_eval.eval_dataset` as the scoring function
* records every step (initial population, parents/children at each iteration,
  population after each iteration, and the final best prompt) into
  ``args.results_json``.
"""

from __future__ import annotations

import functools
import heapq
import json
import os
import random
from typing import Any, Dict, List

import numpy as np
from tqdm import tqdm

from utils import (
    extract_numbers,
    get_final_prompt,
    k_init_pop,
    read_lines,
    setup_log,
)
from llm_client import paraphrase, llm_query
from data.template_ga import templates_2
from data.templates import *  # noqa: F401, F403 - re-exports ``templates``
from dataset_eval import eval_dataset, load_split


class Evoluter:
    def __init__(self, args, llm_config, client):
        self.init_poplulation = []
        self.population: List[str] = []
        self.scores: List[float] = []
        self.marks: List[str] = []
        self.prompts2mark: Dict[str, str] = {}
        self.evaluated_prompts: Dict[str, float] = {}

        self.client, self.llm_config = client, llm_config
        self.public_out_path = args.output

        # Dataset / task resolution. ``args.dataset`` is the new path.
        self.dataset = args.dataset
        self.task = args.dataset or args.task
        self.args = args
        # Metric used everywhere downstream (bert_score / exact_match / f1_mera).
        self.metric = getattr(args, "metric", None)

        os.makedirs(self.public_out_path, exist_ok=True)

        self.logger = logger = setup_log(
            os.path.join(self.public_out_path, "evol.log")
        )
        logger.info("=" * 50)
        logger.info("\n\t" + "\n\t".join(f"{k} = {v}" for k, v in vars(args).items()))
        logger.info("=" * 50)

        self.out_path = os.path.join(self.public_out_path, "dev_result.txt")

        # Load data
        if self.dataset is not None:
            train_examples = load_split(self.dataset, "train")
            val_examples = load_split(self.dataset, "validation")
            # Cap to keep cost manageable
            random.shuffle(train_examples)
            self.dev_data = train_examples[: args.sample_num]
            self.test_data = val_examples[: max(args.test_sample_num, 1)]
            self.task_data = self.dev_data + self.test_data
        else:
            self.task_data = json.load(
                open(f"data/{args.task}.json", "r", encoding="utf-8")
            )["examples"]
            self.dev_data = random.sample(self.task_data, args.sample_num)
            self.test_data = [i for i in self.task_data if i not in self.dev_data]

        # Build the scoring function used everywhere downstream.
        self.eval_func = functools.partial(
            eval_dataset,
            dataset=self.dataset or args.task,
            eval_data=self.dev_data,
            client=client,
            model_index="turbo",
            logger=logger,
            demon=args.demon,
            metric=self.metric,
            **(llm_config or {}),
        )

        # Optimisation history dumped to JSON at the end of every step.
        self.history: Dict[str, Any] = {
            "config": {k: _jsonable(v) for k, v in vars(args).items()},
            "dataset": self.dataset or args.task,
            "metric": self.metric,
            "initial": [],
            "steps": [],
            "final": {},
        }
        self.results_json = args.results_json
        os.makedirs(os.path.dirname(os.path.abspath(self.results_json)) or ".",
                    exist_ok=True)

    # ------------------------------------------------------------------
    # History I/O
    # ------------------------------------------------------------------

    def _save_history(self) -> None:
        try:
            with open(self.results_json, "w", encoding="utf-8") as fh:
                json.dump(self.history, fh, ensure_ascii=False, indent=2)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning(f"Failed to write history JSON: {exc}")

    # ------------------------------------------------------------------
    # Sorting / utility
    # ------------------------------------------------------------------

    def sorted(self):
        best_score = 0
        total_score = 0
        with open(os.path.join(self.public_out_path, "dev_result.txt"), "w") as wf:
            self.scores, self.population, self.marks = (
                list(t)
                for t in zip(
                    *sorted(
                        zip(self.scores, self.population, self.marks),
                        key=lambda x: x[0],
                        reverse=True,
                    )
                )
            )
            for score, prompt, mark in zip(self.scores, self.population, self.marks):
                float_score = float(score)
                if float_score > best_score:
                    best_score = float_score
                total_score += float_score
                wf.write(f"{mark}\t{prompt}\t{score}\n")
            wf.write(f"best score: {best_score}\n")
            wf.write(f"average score: {total_score / max(len(self.scores), 1)}\n")

    def run(self):
        self.evolute()
        self.sorted()

    # ------------------------------------------------------------------
    # Population initialisation
    # ------------------------------------------------------------------

    def init_pop(self):
        args = self.args
        logger = self.logger

        out_path = self.public_out_path
        cur_budget = -1
        cot_cache_path = args.cot_cache_path
        desc_cache_path = args.desc_cache_path

        def load_cache(self, cache_path):
            try:
                cache = json.load(open(cache_path, "r"))
                logger.info(f"---loading prompts from {cache_path}---")
                self.evaluated_prompts = dict(
                    sorted(cache.items(), key=lambda item: item[1], reverse=True)
                )
                init_population = list(self.evaluated_prompts.keys())
            except Exception:
                topk_population = []
                self.evaluated_prompts = {}
                prompt_path = (
                    f"auto_prompts/{args.task}.txt"
                    if args.initial == "ape" and args.task
                    else "prompts.txt"
                )
                pop = read_lines(prompt_path)
                logger.info(
                    "-----evaluating initial population and paraphrasing topk---------"
                )
                for prompt in pop:
                    eval_res = self.eval_func(cot_prompt=prompt)
                    self.evaluated_prompts[prompt] = eval_res
                    topk_population.append((eval_res, prompt))
                topk_population.sort(reverse=True, key=lambda x: x[0])

                with open(cache_path, "w") as wf:
                    self.evaluated_prompts = dict(
                        sorted(self.evaluated_prompts.items(), key=lambda i: i[1])
                    )
                    json.dump(self.evaluated_prompts, wf)
                init_population = [i[1] for i in topk_population]
            return init_population, self.evaluated_prompts

        if args.initial == "ckpt":
            init_population = []
            logger.info(f"------------load from file {args.ckpt_pop}------------")
            ckpt_pop = read_lines(args.ckpt_pop)[: args.popsize]
            for line in ckpt_pop:
                try:
                    mark, prompt, score = line.strip().split("\t")
                    score = float(score)
                except Exception:
                    continue
                self.prompts2mark[prompt] = mark
                self.evaluated_prompts[prompt] = score
                init_population.append(prompt)
                cur_budget = extract_numbers(args.ckpt_pop.split("/")[-1])
            logger.info("current budget: %d" % cur_budget)
        elif args.initial == "cot":
            init_population, self.evaluated_prompts = load_cache(self, cot_cache_path)
            self.prompts2mark = {i: "manual" for i in init_population}
        elif args.initial == "desc":
            init_population, self.evaluated_prompts = load_cache(self, desc_cache_path)
            self.prompts2mark = {i: "ape" for i in init_population}
        elif args.initial == "all":
            init_population_cot, evaluated_cot = load_cache(self, cot_cache_path)
            init_population_desc, evaluated_desc = load_cache(self, desc_cache_path)
            self.evaluated_prompts = {**evaluated_cot, **evaluated_desc}
            self.evaluated_prompts = dict(
                sorted(self.evaluated_prompts.items(),
                       key=lambda item: item[1], reverse=True)
            )
            init_population = list(self.evaluated_prompts.keys())
            self.prompts2mark = {
                i: "manual" if i in init_population_cot else "ape"
                for i in init_population
            }
        else:
            raise ValueError(f"Unsupported --initial {args.initial}")

        # Smoke-test the LLM client.
        _ = paraphrase(
            sentence="Hi, I am a student.",
            type=args.llm_type,
            client=self.client,
            temperature=0.5,
        )
        logger.info("test LLM client success")

        if args.initial_mode in ["para_topk", "para_bottomk", "para_randomk"]:
            k_pop = k_init_pop(args.initial_mode, init_population, k=args.popsize)
            para_population = paraphrase(
                client=self.client,
                sentence=k_pop,
                type=args.llm_type,
                temperature=0.5,
            )
            for i in para_population:
                self.prompts2mark[i] = "para"
            init_population = k_pop + para_population
            init_population = init_population[: args.popsize]
        elif args.initial_mode in ["topk", "bottomk", "randomk"]:
            init_population = k_init_pop(
                args.initial_mode, init_population, k=args.popsize
            )

        cur_best_score = 0
        cur_best_prompt = ""
        total_score = 0

        self.population = list(init_population)
        assert len(self.population) == args.popsize, (
            f"population size {len(self.population)} != popsize {args.popsize}"
        )

        with open(os.path.join(out_path, "step0_pop_para.txt"), "w") as wf:
            for i in self.population:
                if i not in self.evaluated_prompts:
                    self.evaluated_prompts[i] = self.eval_func(cot_prompt=i)
                scores = self.evaluated_prompts[i]
                total_score += scores
                if cur_best_score < scores:
                    cur_best_score = scores
                    cur_best_prompt = i
                wf.write(f"{self.prompts2mark[i]}\t{i}\t{scores}\n")
                self.history["initial"].append({
                    "prompt": i,
                    "mark": self.prompts2mark[i],
                    "score": float(scores),
                })
            wf.write(f"best score: {cur_best_score}\n")
            wf.write(f"average score: {total_score / args.popsize}\n")

        self._save_history()
        return self.evaluated_prompts, cur_budget

    def write_step(self, i, avg_score, best_score):
        out_path = self.public_out_path
        with open(os.path.join(out_path, f"step{i}_pop.txt"), "w") as wf:
            for p in self.population:
                score = self.evaluated_prompts[p]
                wf.write(f"{self.prompts2mark[p]}\t{p}\t{score}\n")
            wf.write(f"best score: {best_score}\n")
            wf.write(f"average score: {avg_score}\n")

    def evolute(self):
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Final test set evaluation
    # ------------------------------------------------------------------

    def test(self, step):
        self.logger.info(f"----------testing step {step} population----------")
        pop_marks = [self.prompts2mark[i] for i in self.population]
        pop_scores = [self.evaluated_prompts[i] for i in self.population]
        self.population, pop_scores, pop_marks = (
            list(t)
            for t in zip(
                *sorted(
                    zip(self.population, pop_scores, pop_marks),
                    key=lambda x: x[1],
                    reverse=True,
                )
            )
        )

        test_prompt_num = max(self.args.popsize // 2, 1)
        test_results: List[Dict[str, Any]] = []
        with open(
            os.path.join(self.public_out_path, f"step{step}_pop_test.txt"), "w"
        ) as wf:
            for i in tqdm(range(test_prompt_num)):
                test_prompt = self.population[i]
                test_mark = pop_marks[i]
                test_score = eval_dataset(
                    dataset=self.dataset or self.args.task,
                    cot_prompt=test_prompt,
                    eval_data=self.test_data,
                    client=self.client,
                    logger=self.logger,
                    demon=self.args.demon,
                    metric=self.metric,
                )
                dev_score = self.evaluated_prompts[test_prompt]
                wf.write(f"{test_mark}\t{test_prompt}\t{dev_score}\t{test_score}\n")
                wf.flush()
                test_results.append({
                    "prompt": test_prompt,
                    "mark": test_mark,
                    "dev_score": float(dev_score),
                    "test_score": float(test_score),
                })

        best = max(test_results, key=lambda r: r["test_score"]) if test_results else {}
        self.history["final"] = {
            "dataset": self.dataset or self.args.task,
            "metric": self.metric,
            "best_prompt": best.get("prompt", ""),
            "dev_score": best.get("dev_score", 0.0),
            "test_score": best.get("test_score", 0.0),
            "top_candidates": test_results,
        }
        self._save_history()


# ---------------------------------------------------------------------------
# Differential Evolution
# ---------------------------------------------------------------------------

class DEEvoluter(Evoluter):
    def __init__(self, args, llm_config, client):
        super().__init__(args, llm_config=llm_config, client=client)
        self.template = templates[args.template]["sim"]  # type: ignore[name-defined]

    def evolute(self):
        logger = self.logger
        args = self.args
        self.evaluated_prompts, cur_budget = self.init_pop()
        template = self.template
        best_scores: List[float] = []
        avg_scores: List[float] = []

        cur_best_prompt, cur_best_score = max(
            self.evaluated_prompts.items(), key=lambda x: x[1]
        )

        for step in range(cur_budget + 1, args.budget):
            logger.info(f"step: {step}")
            new_pop: List[str] = []
            total_score = 0.0
            best_score = 0.0
            step_events: List[Dict[str, Any]] = []

            for j in range(args.popsize):
                logger.info(f"step {step}, pop {j}")
                old_prompt = self.population[j]
                if old_prompt not in self.evaluated_prompts:
                    self.evaluated_prompts[old_prompt] = self.eval_func(
                        cot_prompt=old_prompt
                    )
                old_scores = self.evaluated_prompts[old_prompt]

                cur_candidates = {
                    old_prompt: {
                        "score": old_scores,
                        "mark": self.prompts2mark[old_prompt],
                    },
                }

                candidates = [self.population[k]
                              for k in range(args.popsize) if k != j]
                a, b, c = np.random.choice(candidates, 3, replace=False)
                if not args.donor_random:
                    c = cur_best_prompt
                request_content = (
                    template.replace("<prompt0>", old_prompt)
                    .replace("<prompt1>", a)
                    .replace("<prompt2>", b)
                    .replace("<prompt3>", c)
                )
                logger.info(f"evolution example:\n{request_content}")

                de_prompt = llm_query(
                    client=self.client,
                    data=request_content,
                    type=args.llm_type,
                    task=False,
                    temperature=0.5,
                )
                de_prompt = get_final_prompt(de_prompt)
                logger.info(f"de prompt: {de_prompt}")

                de_eval_res = self.eval_func(cot_prompt=de_prompt)
                logger.info(f"de_score: {de_eval_res}")
                self.prompts2mark[de_prompt] = "evoluted"
                cur_candidates[de_prompt] = {
                    "score": de_eval_res,
                    "mark": self.prompts2mark[de_prompt],
                }
                self.evaluated_prompts[de_prompt] = de_eval_res

                selected_prompt = max(
                    cur_candidates, key=lambda x: cur_candidates[x]["score"]
                )
                selected_score = float(cur_candidates[selected_prompt]["score"])
                selected_mark = cur_candidates[selected_prompt]["mark"]
                total_score += selected_score
                if selected_score > best_score:
                    best_score = selected_score
                    if best_score > cur_best_score:
                        cur_best_score = best_score
                        cur_best_prompt = selected_prompt

                new_pop.append(selected_prompt)
                step_events.append({
                    "j": j,
                    "old_prompt": old_prompt,
                    "old_score": float(old_scores),
                    "parents": [str(a), str(b), str(c)],
                    "child_prompt": de_prompt,
                    "child_score": float(de_eval_res),
                    "selected_prompt": selected_prompt,
                    "selected_score": selected_score,
                    "selected_mark": selected_mark,
                })

            avg_score = total_score / args.popsize
            avg_scores.append(avg_score)
            best_scores.append(best_score)
            self.population = new_pop

            self.write_step(i=step, best_score=best_score, avg_score=avg_score)
            self.history["steps"].append({
                "step": step,
                "best_score": float(best_score),
                "avg_score": float(avg_score),
                "events": step_events,
                "population": [
                    {"prompt": p,
                     "score": float(self.evaluated_prompts[p]),
                     "mark": self.prompts2mark[p]}
                    for p in self.population
                ],
                "running_best": {
                    "prompt": cur_best_prompt,
                    "score": float(cur_best_score),
                },
            })
            self._save_history()

        self.test(step=args.budget - 1)

        logger.info(f"best_scores: {','.join(str(s) for s in best_scores)}")
        logger.info(f"avg_scores: {','.join(f'{s:.4f}' for s in avg_scores)}")
        self.scores = [self.evaluated_prompts[i] for i in self.population]
        self.marks = [self.prompts2mark[i] for i in self.population]
        self.sorted()


# ---------------------------------------------------------------------------
# Genetic Algorithm
# ---------------------------------------------------------------------------

class GAEvoluter(Evoluter):
    def __init__(self, args, llm_config, client):
        super().__init__(args, llm_config=llm_config, client=client)
        self.template = templates_2["sim"]

    def evolute(self):
        logger = self.logger
        args = self.args
        self.evaluated_prompts, cur_budget = self.init_pop()
        template = self.template

        best_scores: List[float] = []
        avg_scores: List[float] = []

        for step in range(cur_budget + 1, args.budget):
            total_score = 0.0
            best_score = 0.0
            fitness = np.array([self.evaluated_prompts[i] for i in self.population])
            new_pop: List[str] = []
            step_events: List[Dict[str, Any]] = []

            if args.sel_mode == "wheel":
                fsum = fitness.sum() if fitness.sum() > 0 else 1.0
                wheel_idx = np.random.choice(
                    np.arange(args.popsize),
                    size=args.popsize,
                    replace=True,
                    p=fitness / fsum,
                ).tolist()
                parent_pop = [self.population[i] for i in wheel_idx]
            elif args.sel_mode in ["random", "tour"]:
                parent_pop = list(self.population)
            else:
                raise ValueError(f"Unknown sel_mode {args.sel_mode}")

            for j in range(args.popsize):
                if args.sel_mode in ["random", "wheel"]:
                    parents = random.sample(parent_pop, 2)
                    cand_a, cand_b = parents
                else:  # tour
                    group_a = random.sample(parent_pop, 2)
                    group_b = random.sample(parent_pop, 2)
                    cand_a = max(group_a, key=lambda x: self.evaluated_prompts[x])
                    cand_b = max(group_b, key=lambda x: self.evaluated_prompts[x])

                request_content = (
                    template.replace("<prompt1>", cand_a)
                    .replace("<prompt2>", cand_b)
                )
                logger.info(f"evolution example:\n{request_content}")
                child_prompt = llm_query(
                    client=self.client,
                    data=request_content,
                    type=args.llm_type,
                    task=False,
                    temperature=0.5,
                )
                child_prompt = get_final_prompt(child_prompt)
                logger.info(f"child prompt: {child_prompt}")

                de_eval_res = self.eval_func(cot_prompt=child_prompt)
                logger.info(f"new score: {de_eval_res}")
                self.prompts2mark[child_prompt] = "evoluted"
                self.evaluated_prompts[child_prompt] = de_eval_res

                selected_prompt = child_prompt
                selected_score = de_eval_res
                if args.ga_mode == "std":
                    self.population[j] = selected_prompt

                new_pop.append(selected_prompt)
                total_score += selected_score
                if selected_score > best_score:
                    best_score = selected_score

                step_events.append({
                    "j": j,
                    "parents": [cand_a, cand_b],
                    "child_prompt": child_prompt,
                    "child_score": float(de_eval_res),
                    "selected_prompt": selected_prompt,
                    "selected_score": float(selected_score),
                })

            if args.ga_mode == "topk":
                double_pop = list(set(self.population + new_pop))
                double_pop = sorted(
                    double_pop, key=lambda x: self.evaluated_prompts[x], reverse=True
                )
                self.population = double_pop[: args.popsize]
                total_score = sum(self.evaluated_prompts[i] for i in self.population)
                best_score = self.evaluated_prompts[self.population[0]]

            avg_score = total_score / args.popsize
            avg_scores.append(avg_score)
            best_scores.append(best_score)

            self.write_step(i=step, best_score=best_score, avg_score=avg_score)
            self.history["steps"].append({
                "step": step,
                "best_score": float(best_score),
                "avg_score": float(avg_score),
                "events": step_events,
                "population": [
                    {"prompt": p,
                     "score": float(self.evaluated_prompts[p]),
                     "mark": self.prompts2mark.get(p, "")}
                    for p in self.population
                ],
            })
            self._save_history()

            if step == args.budget - 1:
                self.test(step=step)

        logger.info(f"best_scores: {','.join(str(s) for s in best_scores)}")
        logger.info(f"avg_scores: {','.join(f'{s:.4f}' for s in avg_scores)}")
        self.scores = [self.evaluated_prompts[i] for i in self.population]
        self.marks = [self.prompts2mark[i] for i in self.population]
        self.sorted()


# ---------------------------------------------------------------------------
# APE-style paraphrasing
# ---------------------------------------------------------------------------

class ParaEvoluter(Evoluter):
    def __init__(self, args, llm_config, client):
        super().__init__(args, llm_config=llm_config, client=client)

    def init_pop(self):
        args = self.args
        logger = self.logger
        # Use prompts.txt as the seed population (BBH auto_prompts don't apply
        # to the generic datasets).
        seed_prompts_path = (
            f"./auto_prompts/{args.task}.txt"
            if args.task and not args.dataset else
            "prompts.txt"
        )
        self.init_population = read_lines(seed_prompts_path)[: args.popsize]
        self.prompts2mark = {i: "ape" for i in self.init_population}
        logger.info("initial population:")
        for i in self.init_population:
            logger.info(i)
        with open(f"{self.public_out_path}/init.txt", "w") as wf:
            for i in self.init_population:
                wf.write(f"{i}\n")

    def evolute(self):
        self.init_pop()
        args = self.args
        k = args.popsize
        logger = self.logger
        self.evaluated_prompts = {}
        cur_budget = -1
        topk_heap: List = []
        best_scores: List[float] = []
        avg_scores: List[float] = []

        _ = paraphrase(
            sentence=self.init_population[0],
            client=self.client,
            type=args.llm_type,
        )

        for prompt in self.init_population:
            score = self.eval_func(cot_prompt=prompt)
            self.evaluated_prompts[prompt] = score
            self.logger.info(f"{self.prompts2mark[prompt]}: {prompt}, {score}")
            heapq.heappush(topk_heap, (score, prompt))
            self.history["initial"].append({
                "prompt": prompt,
                "mark": self.prompts2mark[prompt],
                "score": float(score),
            })
        self._save_history()

        for step in range(cur_budget + 1, args.budget):
            best_score = 0.0
            total_score = 0.0
            self.population, self.marks, self.scores = [], [], []
            logger.info(f"step: {step}")
            top_k = heapq.nlargest(k, topk_heap)

            new_prompts: List = []
            step_events: List[Dict[str, Any]] = []
            paraphrased_prompts = paraphrase(
                sentence=[i[1] for i in top_k],
                client=self.client,
                type=args.llm_type,
                temperature=0.5,
            )
            if isinstance(paraphrased_prompts, str):
                paraphrased_prompts = [paraphrased_prompts]

            for i, prompt in enumerate(paraphrased_prompts):
                logger.info(f"step: {step}, prompt: {prompt}")
                new_score = self.eval_func(cot_prompt=prompt)
                self.prompts2mark[prompt] = "para"
                logger.info(f"paraphrased: {prompt}, {new_score}")
                new_prompts.append((new_score, prompt))
                self.evaluated_prompts[prompt] = new_score
                step_events.append({
                    "parent": top_k[i][1],
                    "parent_score": float(top_k[i][0]),
                    "child_prompt": prompt,
                    "child_score": float(new_score),
                })

            for new_prompt in new_prompts:
                heapq.heappushpop(topk_heap, new_prompt)

            for _, prompt in topk_heap:
                self.population.append(prompt)
                cur_score = float(self.evaluated_prompts[prompt])
                if best_score < cur_score:
                    best_score = cur_score
                total_score += cur_score
                mark = "manual" if prompt in self.init_population else "para"
                self.marks.append(mark)
            avg_score = total_score / max(len(topk_heap), 1)
            best_scores.append(best_score)
            avg_scores.append(avg_score)

            self.write_step(i=step, best_score=best_score, avg_score=avg_score)
            self.history["steps"].append({
                "step": step,
                "best_score": float(best_score),
                "avg_score": float(avg_score),
                "events": step_events,
                "population": [
                    {"prompt": p,
                     "score": float(self.evaluated_prompts[p]),
                     "mark": self.prompts2mark.get(p, "")}
                    for p in self.population
                ],
            })
            self._save_history()

            if step == args.budget - 1:
                self.test(step=step)

        logger.info(f"best_scores: {','.join(str(s) for s in best_scores)}")
        logger.info(f"avg_scores: {','.join(f'{s:.4f}' for s in avg_scores)}")
        self.scores = [self.evaluated_prompts[i] for i in self.population]
        self.marks = [self.prompts2mark[i] for i in self.population]
        self.sorted()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _jsonable(value: Any) -> Any:
    """Convert argparse Namespace values to something JSON serialisable."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)
