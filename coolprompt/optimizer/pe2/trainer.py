"""PE2 trainer: beam-search loop for prompt optimization."""

import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.messages.ai import AIMessage

from coolprompt.evaluator import Evaluator
from coolprompt.evaluator.metrics import ClassificationMetric
from coolprompt.optimizer.pe2.node import Node
from coolprompt.optimizer.pe2.proposer import Proposer
from coolprompt.utils.enums import Task
from coolprompt.utils.logging_config import logger
from coolprompt.utils.parsing import extract_answer


class PE2Trainer:
    """Core beam-search trainer for PE2 prompt optimization.

    Args:
        model (BaseLanguageModel): LLM used for inference.
        evaluator (Evaluator): Evaluator for scoring prompts.
        proposer (Proposer): Proposer for generating refined prompts.
        train_dataset (List[str]): Training input samples.
        train_targets (List[str]): Training ground-truth targets.
        val_dataset (List[str]): Validation input samples.
        val_targets (List[str]): Validation ground-truth targets.
        template (str): Prompt template for the task.
        train_steps (int): Number of beam-search iterations.
        n_beam (int): Beam width (top-k nodes kept per step).
        n_expand (int): Number of children per selected node.
        batch_size (int): Number of failure examples shown to proposer.
        backtrack (bool): If True, select best across all timesteps.
    """

    ANS_TAGS = ("<ans>", "</ans>")

    def __init__(
        self,
        model: BaseLanguageModel,
        evaluator: Evaluator,
        proposer: Proposer,
        train_dataset: List[str],
        train_targets: List[str],
        val_dataset: List[str],
        val_targets: List[str],
        template: str,
        train_steps: int = 3,
        n_beam: int = 3,
        n_expand: int = 4,
        batch_size: int = 4,
        backtrack: bool = True,
        feedback_mode: str = "auto",
    ) -> None:
        self.model = model
        self.evaluator = evaluator
        self.proposer = proposer
        self.train_dataset = train_dataset
        self.train_targets = train_targets
        self.val_dataset = val_dataset
        self.val_targets = val_targets
        self.template = template
        self.train_steps = train_steps
        self.n_beam = n_beam
        self.n_expand = n_expand
        self.batch_size = batch_size
        self.backtrack = backtrack
        self.feedback_mode = feedback_mode
        self._node_counter = 0

    def _next_id(self) -> int:
        """Returns the next unique node ID."""
        nid = self._node_counter
        self._node_counter += 1
        return nid

    def train(self, initial_prompt: str) -> str:
        """Runs the PE2 beam-search optimization.

        Args:
            initial_prompt (str): The starting prompt to optimize.

        Returns:
            str: The best prompt found during optimization.
        """
        initial_node = Node(
            timestamp=0,
            id=self._next_id(),
            prompt=initial_prompt,
        )
        states: List[List[Node]] = [[initial_node]]

        for t in range(self.train_steps):
            logger.info(f"PE2 step {t + 1}/{self.train_steps}")

            # Evaluate nodes in the latest state on validation set
            for node in states[-1]:
                if "val" not in node.scores:
                    score = self.evaluator.evaluate(
                        prompt=node.prompt,
                        dataset=self.val_dataset,
                        targets=self.val_targets,
                        template=self.template,
                    )
                    node.register_score(score, "val")
                    logger.info(
                        f"  Node {node.id} val score: {score:.4f}"
                    )

            # Select top n_beam nodes
            candidates = self._select_candidates(states)
            best_score = max(n.scores["val"] for n in candidates)
            logger.info(
                f"  Best val score at step {t + 1}: {best_score:.4f}"
            )

            # If last step, just return the best
            if t == self.train_steps - 1:
                break

            new_nodes: List[Node] = []
            # Collect all proposal jobs
            proposal_jobs = []
            for node in candidates:
                # Get per-example results on training set
                results = self._get_per_example_results(
                    node.prompt
                )
                failures = [
                    r for r in results if not r["correct"]
                ]

                if not failures:
                    logger.info(
                        f"  Node {node.id}: no failures "
                        "on train set, skipping expansion"
                    )
                    continue

                cfb = None
                if self.feedback_mode != "off":
                    metric = getattr(
                        self.evaluator, "metric", None
                    )
                    if metric is not None and hasattr(
                        metric, "failure_breakdown"
                    ):
                        cfb = metric.failure_breakdown(
                            [
                                f["raw_output"]
                                for f in failures
                            ],
                            [f["target"] for f in failures],
                        )

                full_template = self._instantiate_template(
                    node.prompt
                )

                for _ in range(self.n_expand):
                    sampled = self._sample_failures(
                        failures, self.batch_size
                    )
                    examples_str = self._pack_examples(
                        sampled
                    )
                    proposal_jobs.append(
                        (node, examples_str,
                         full_template, len(sampled),
                         best_score, cfb)
                    )

            if not proposal_jobs:
                logger.info(
                    "  No proposal jobs, stopping"
                )
                break

            # Run proposals in parallel
            def _do_propose(job):
                n, ex_str, ft, bs, bvs, cf = job
                prompt, _ = self.proposer.propose(
                    node=n,
                    examples_str=ex_str,
                    full_template=ft,
                    batch_size=bs,
                    best_val_score=bvs,
                    constraint_feedback=cf,
                )
                return n, prompt

            with ThreadPoolExecutor(
                max_workers=min(len(proposal_jobs), 12)
            ) as pool:
                futures = [
                    pool.submit(_do_propose, j)
                    for j in proposal_jobs
                ]
                for fut in as_completed(futures):
                    node, new_prompt = fut.result()
                    child = Node(
                        timestamp=t + 1,
                        id=self._next_id(),
                        prompt=new_prompt,
                        parent=node.id,
                    )
                    node.n_child += 1
                    new_nodes.append(child)

            # Deduplicate against all existing nodes
            all_existing = [n for state in states for n in state]
            new_nodes = self._deduplicate(new_nodes, all_existing)

            if not new_nodes:
                logger.info("  No new unique nodes generated, stopping")
                break

            states.append(new_nodes)
            logger.info(f"  Generated {len(new_nodes)} new nodes")

        # Return the best prompt by val score
        all_nodes = [n for state in states for n in state]
        scored = [n for n in all_nodes if "val" in n.scores]
        best = max(scored, key=lambda n: n.scores["val"])
        logger.info(
            f"PE2 best prompt (node {best.id}, "
            f"val={best.scores['val']:.4f})"
        )
        return best.prompt

    def _select_candidates(
        self, states: List[List[Node]]
    ) -> List[Node]:
        """Selects top n_beam nodes by validation score.

        Args:
            states (List[List[Node]]): All beam states so far.

        Returns:
            List[Node]: Top-k nodes by val score.
        """
        if self.backtrack:
            pool = [n for state in states for n in state]
        else:
            pool = list(states[-1])

        scored = [n for n in pool if "val" in n.scores]
        scored.sort(key=lambda n: n.scores["val"], reverse=True)
        return scored[: self.n_beam]

    def _instantiate_template(self, prompt: str) -> str:
        """Instantiates the task template with the prompt.

        Replaces the prompt placeholder with the actual prompt
        and the input placeholder with a literal ``<input>``,
        matching the official PE2 behavior where the proposer
        sees the concrete full prompt structure.

        Args:
            prompt (str): The instruction prompt.

        Returns:
            str: The instantiated full template string.
        """
        # Escape braces in prompt to avoid format errors
        safe_prompt = prompt.replace("{", "{{").replace(
            "}", "}}"
        )
        if self.evaluator.task == Task.CLASSIFICATION:
            if isinstance(
                self.evaluator.metric, ClassificationMetric
            ) and self.evaluator.metric.label_to_id:
                labels = ", ".join(
                    map(
                        str,
                        self.evaluator.metric.label_to_id
                        .keys(),
                    )
                )
            else:
                labels = "<labels>"
            return self.template.format(
                PROMPT=safe_prompt,
                LABELS=labels,
                INPUT="<input>",
            )
        else:
            return self.template.format(
                PROMPT=safe_prompt,
                INPUT="<input>",
            )

    def _get_per_example_results(
        self, prompt: str
    ) -> List[dict]:
        """Runs the model on training data and checks
        correctness via exact match.

        Uses exact string comparison on extracted answers
        for failure detection, matching official PE2 behavior
        (``score == 0.0`` where score is EM). The actual
        evaluation metric is only used for overall prompt
        scoring on the validation set.

        Args:
            prompt (str): The prompt to evaluate.

        Returns:
            List[dict]: Per-example results with keys: input,
                target, answer, raw_output, correct.
        """
        if self.evaluator.task == Task.CLASSIFICATION:
            if isinstance(
                self.evaluator.metric, ClassificationMetric
            ):
                self.evaluator.metric.extract_labels(
                    self.train_targets
                )

        full_prompts = [
            self.evaluator._get_full_prompt(
                prompt, sample, self.template
            )
            for sample in self.train_dataset
        ]
        raw_answers = self.model.batch(full_prompts)
        raw_answers = [
            a.content if isinstance(a, AIMessage) else a
            for a in raw_answers
        ]

        results = []
        for inp, target, raw in zip(
            self.train_dataset,
            self.train_targets,
            raw_answers,
        ):
            raw_str = str(raw).strip()
            extracted = extract_answer(
                raw_str, self.ANS_TAGS, raw_str
            )
            extracted_str = str(extracted).strip()
            target_str = str(target).strip()

            correct = (
                extracted_str.lower() == target_str.lower()
            )
            results.append({
                "input": inp,
                "target": target_str,
                "answer": extracted_str,
                "raw_output": raw_str,
                "correct": correct,
            })
        return results

    def _pack_examples(self, examples: List[dict]) -> str:
        """Formats failure examples matching official PE2 format.

        Includes a Reasoning field when the raw model output
        contains more than the extracted answer (e.g. chain-of-
        thought), matching official PE2's inclusion of reasoning
        traces for math/BBH tasks.

        Args:
            examples (List[dict]): List of failure example dicts.

        Returns:
            str: Formatted string of numbered examples.
        """
        parts = []
        for i, ex in enumerate(examples, 1):
            lines = [
                f"### Example {i}",
                f"Input: {ex['input']}",
            ]
            # Include reasoning when raw output differs from
            # the extracted answer (indicates CoT / reasoning)
            raw = ex.get("raw_output", "")
            if raw and raw != ex["answer"]:
                lines.append(f"Reasoning: {raw}")
            lines.append(f"Output: {ex['answer']}")
            lines.append(f"Label: {ex['target']}")
            parts.append("\n".join(lines))
        return "\n\n".join(parts)

    def _deduplicate(
        self,
        new_nodes: List[Node],
        all_nodes: List[Node],
    ) -> List[Node]:
        """Removes nodes whose prompts already exist.

        Args:
            new_nodes (List[Node]): Candidate new nodes.
            all_nodes (List[Node]): All existing nodes.

        Returns:
            List[Node]: Deduplicated new nodes.
        """
        existing = {n.prompt.strip().lower() for n in all_nodes}
        unique = []
        for node in new_nodes:
            key = node.prompt.strip().lower()
            if key not in existing:
                existing.add(key)
                unique.append(node)
        return unique

    def _sample_failures(
        self, results: List[dict], k: int
    ) -> List[dict]:
        """Randomly samples k failure examples.

        Args:
            results (List[dict]): List of failure example dicts.
            k (int): Number of examples to sample.

        Returns:
            List[dict]: Sampled failure examples.
        """
        return random.sample(results, min(k, len(results)))
