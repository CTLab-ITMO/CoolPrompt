"""OPRO trainer: beam-search with trajectory updates."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from coolprompt.optimizer.pe2.node import Node
from coolprompt.optimizer.pe2.trainer import PE2Trainer
from coolprompt.optimizer.opro.proposer import OPROProposer
from coolprompt.utils.logging_config import logger


class OPROTrainer(PE2Trainer):
    """Extends PE2Trainer to feed trajectory to OPROProposer.

    After evaluating each node, updates the proposer's
    trajectory. Also expands nodes even when there are no
    training failures, since OPRO does not use failure
    examples.
    """

    def train(self, initial_prompt: str) -> str:
        """Runs OPRO beam-search optimization.

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
        states: List[List[Node]] = [[initial_node]]

        for t in range(self.train_steps):
            logger.info(f"OPRO step {t + 1}/{self.train_steps}")

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
                        f"  Node {node.id} val score: "
                        f"{score:.4f}"
                    )
                    # Update OPRO trajectory
                    assert isinstance(
                        self.proposer, OPROProposer
                    )
                    self.proposer.update_trajectory(
                        node.prompt, score
                    )

            candidates = self._select_candidates(states)
            best_score = max(
                n.scores["val"] for n in candidates
            )
            logger.info(
                f"  Best val score at step {t + 1}: "
                f"{best_score:.4f}"
            )

            if t == self.train_steps - 1:
                break

            new_nodes: List[Node] = []
            proposal_jobs = [
                node
                for node in candidates
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

            all_existing = [
                n for state in states for n in state
            ]
            new_nodes = self._deduplicate(
                new_nodes, all_existing
            )

            if not new_nodes:
                logger.info(
                    "  No new unique nodes generated, stopping"
                )
                break

            states.append(new_nodes)
            logger.info(
                f"  Generated {len(new_nodes)} new nodes"
            )

        all_nodes = [n for state in states for n in state]
        scored = [n for n in all_nodes if "val" in n.scores]
        best = max(scored, key=lambda n: n.scores["val"])
        logger.info(
            f"OPRO best prompt (node {best.id}, "
            f"val={best.scores['val']:.4f})"
        )
        return best.prompt
