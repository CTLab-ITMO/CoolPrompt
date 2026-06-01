"""APE trainer: evaluate-select-paraphrase loop."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from langchain_core.language_models.base import BaseLanguageModel

from coolprompt.evaluator import Evaluator
from coolprompt.optimizer.ape.proposer import APEProposer
from coolprompt.optimizer.pe2.node import Node
from coolprompt.utils.logging_config import logger


class APETrainer:
    """APE trainer with its own evaluate-select-paraphrase loop.

    Matches the original APE paper: evaluates candidates on
    validation data, selects top-k, paraphrases each to
    generate new candidates, and repeats.

    No failure mining or training data needed — the proposer
    simply paraphrases the current prompt.

    Args:
        model (BaseLanguageModel): LLM used for inference.
        evaluator (Evaluator): Evaluator for scoring prompts.
        proposer (APEProposer): Proposer for paraphrasing.
        val_dataset (List[str]): Validation input samples.
        val_targets (List[str]): Validation ground-truth
            targets.
        template (str): Prompt template for the task.
        train_steps (int): Number of optimization iterations.
        n_beam (int): Top-k candidates to keep per step.
        n_expand (int): Paraphrases per selected candidate.
    """

    def __init__(
        self,
        model: BaseLanguageModel,
        evaluator: Evaluator,
        proposer: APEProposer,
        val_dataset: List[str],
        val_targets: List[str],
        template: str,
        train_steps: int = 3,
        n_beam: int = 3,
        n_expand: int = 4,
    ) -> None:
        self.model = model
        self.evaluator = evaluator
        self.proposer = proposer
        self.val_dataset = val_dataset
        self.val_targets = val_targets
        self.template = template
        self.train_steps = train_steps
        self.n_beam = n_beam
        self.n_expand = n_expand
        self._node_counter = 0

    def _next_id(self) -> int:
        """Returns the next unique node ID."""
        nid = self._node_counter
        self._node_counter += 1
        return nid

    def train(self, initial_prompt: str) -> str:
        """Runs the APE evaluate-select-paraphrase loop.

        Args:
            initial_prompt (str): The starting prompt.

        Returns:
            str: The best prompt found during optimization.
        """
        initial_node = Node(
            timestamp=0,
            id=self._next_id(),
            prompt=initial_prompt,
        )
        all_nodes: List[Node] = [initial_node]
        unevaluated: List[Node] = [initial_node]

        for t in range(self.train_steps):
            logger.info(
                f"APE step {t + 1}/{self.train_steps}"
            )

            for node in unevaluated:
                score = self.evaluator.evaluate(
                    prompt=node.prompt,
                    dataset=self.val_dataset,
                    targets=self.val_targets,
                    template=self.template,
                )
                node.register_score(score, "val")
                logger.info(
                    f"  Node {node.id} val score: "
                    f"{score:.4f}"
                )

            scored = [
                n for n in all_nodes
                if "val" in n.scores
            ]
            scored.sort(
                key=lambda n: n.scores["val"], reverse=True
            )
            selected = scored[: self.n_beam]
            best_score = selected[0].scores["val"]
            logger.info(
                f"  Best val score at step {t + 1}: "
                f"{best_score:.4f}"
            )

            if t == self.train_steps - 1:
                break

            proposal_jobs = [
                node
                for node in selected
                for _ in range(self.n_expand)
            ]

            def _do_propose(n):
                prompt, _ = self.proposer.propose(
                    node=n,
                    examples_str="",
                    full_template=self.template,
                    batch_size=0,
                )
                return n, prompt

            new_nodes: List[Node] = []
            with ThreadPoolExecutor(
                max_workers=min(len(proposal_jobs), 12)
            ) as pool:
                futures = [
                    pool.submit(_do_propose, j)
                    for j in proposal_jobs
                ]
                for fut in as_completed(futures):
                    parent, new_prompt = fut.result()
                    child = Node(
                        timestamp=t + 1,
                        id=self._next_id(),
                        prompt=new_prompt,
                        parent=parent.id,
                    )
                    parent.n_child += 1
                    new_nodes.append(child)

            existing = {
                n.prompt.strip().lower() for n in all_nodes
            }
            unique = []
            for node in new_nodes:
                key = node.prompt.strip().lower()
                if key not in existing:
                    existing.add(key)
                    unique.append(node)

            if not unique:
                logger.info(
                    "  No new unique nodes generated, "
                    "stopping"
                )
                break

            all_nodes.extend(unique)
            unevaluated = unique
            logger.info(
                f"  Generated {len(unique)} new candidates"
            )

        scored_all = [
            n for n in all_nodes if "val" in n.scores
        ]
        best = max(
            scored_all, key=lambda n: n.scores["val"]
        )
        logger.info(
            f"APE best prompt (node {best.id}, "
            f"val={best.scores['val']:.4f})"
        )
        return best.prompt
