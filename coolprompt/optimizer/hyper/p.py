class HyPEROptimizer(Optimizer):
    """Experimental compact HyPER optimizer variant."""

    def __init__(
        self,
        model: Any,
        evaluator: Evaluator,
        *,
        n_iterations: int = 5,
        patience: int = None,
        n_candidates: int = 3,
        top_n_candidates: int = 3,
        k_samples: int = 3,
        mini_batch_size: int = 16,
        contrastive_probability: float = 0.5,
        contrastive_max_answer_chars: int = 500,
        feedback_answer_head_chars: int = 500,
        feedback_answer_tail_chars: int = 500,
        enable_instance_leak_audit: bool = True,
        random_seed: Optional[int] = None,
    ) -> None:
        """Initialize compact HyPER search parameters."""
        super().__init__(model)
        self.meta_prompt_module = MetaPromptOptimizer(model)
        self.evaluator = evaluator
        self.enable_instance_leak_audit = enable_instance_leak_audit
        self.feedback_module = FeedbackModule(
            model,
            section_specs=self.meta_prompt_module.builder.config.section_specs,
            contrastive_probability=contrastive_probability,
            contrastive_max_answer_chars=contrastive_max_answer_chars,
            feedback_answer_head_chars=feedback_answer_head_chars,
            feedback_answer_tail_chars=feedback_answer_tail_chars,
        )
        self.n_iterations = n_iterations
        self.patience = patience
        self.n_candidates = n_candidates
        self.top_n_candidates = top_n_candidates
        self.k_samples = k_samples
        self.mini_batch_size = mini_batch_size
        self.random_seed = random_seed

    def _get_variants_from_best(self, best_prompt: str, n_candidates: int) -> List[str]:
        raw_result = get_model_answer_extracted(
            self.model,
            PARAPHRASE_PROMPT.format(prompt=best_prompt),
            n=n_candidates,
            temperature=0.9,
        )
        return [best_prompt] + [self._process_model_output(r) for r in raw_result]

    def _process_model_output(self, output: Any) -> str:
        return output if isinstance(output, str) else str(output)

    def _build_feedback_sources(
        self,
        selected: List[Tuple[str, EvalResultDetailed]],
        candidates: List[str],
        results: List[EvalResultDetailed],
    ) -> List[Tuple[str, EvalResultDetailed]]:
        selected_prompts = {candidate for candidate, _ in selected}
        feedback_sources = [
            (candidate, result)
            for candidate, result in selected
            if result.failed_examples
        ]

        substitutes_needed = self.top_n_candidates - len(feedback_sources)
        if substitutes_needed <= 0:
            return feedback_sources

        substitutes = sorted(
            [
                (candidate, result)
                for candidate, result in zip(candidates, results)
                if candidate not in selected_prompts and result.failed_examples
            ],
            key=lambda item: item[1].aggregate_score
            if item[1].aggregate_score is not None
            else 0.0,
            reverse=True,
        )[:substitutes_needed]

        return feedback_sources + substitutes

    def _build_contrastive_examples(
        self,
        failed_sample: List[Any],
        source_prompt: str,
        candidates: List[str],
        results: List[EvalResultDetailed],
    ) -> List[List[ContrastiveCandidate]]:
        contrastive_per_failure: List[List[ContrastiveCandidate]] = []

        for failed_example in failed_sample:
            alternatives: List[ContrastiveCandidate] = []
            if failed_example.batch_index >= 0:
                for other_prompt, other_result in zip(candidates, results):
                    if other_prompt == source_prompt:
                        continue
                    if not other_result.score_per_task or not other_result.raw_outputs:
                        continue
                    if failed_example.batch_index >= len(other_result.score_per_task):
                        continue

                    raw_answer = other_result.raw_outputs[failed_example.batch_index]
                    alternatives.append(
                        ContrastiveCandidate(
                            prompt=other_prompt,
                            score=float(other_result.score_per_task[failed_example.batch_index]),
                            raw_answer=raw_answer,
                            parsed_answer=self.evaluator.metric.parse_output(raw_answer),
                        )
                    )
            contrastive_per_failure.append(alternatives)

        return contrastive_per_failure

    def _generate_recommendations(
        self,
        selected: List[Tuple[str, EvalResultDetailed]],
        candidates: List[str],
        results: List[EvalResultDetailed],
        meta_info: Optional[dict[str, Any]],
    ) -> List[Recommendation]:
        recommendations: List[Recommendation] = []
        feedback_sources = self._build_feedback_sources(selected, candidates, results)

        for candidate_prompt, result in feedback_sources:
            failed_sample = random.sample(
                result.failed_examples,
                min(self.k_samples, len(result.failed_examples)),
            )
            contrastive_examples = self._build_contrastive_examples(
                failed_sample,
                candidate_prompt,
                candidates,
                results,
            )
            recommendations.extend(
                self.feedback_module.generate_recommendations(
                    candidate_prompt,
                    failed_sample,
                    contrastive_candidates_per_failure=contrastive_examples,
                )
            )

        if not recommendations:
            return []

        recommendations = self.feedback_module.filter_recommendations(recommendations)
        problem_description = (meta_info or {}).get("problem_description", "")

        if self.enable_instance_leak_audit and problem_description:
            recommendations = self.feedback_module.drop_instance_leaks(
                recommendations,
                problem_description,
            )

        return recommendations

    def optimize(
        self,
        prompt: str,
        dataset_split: Tuple[
            Sequence[str], Sequence[str], Sequence[str], Sequence[str]
        ],
        meta_info: Optional[dict[str, Any]] = None,
    ) -> str:
        """Run compact HyPER optimization and return the best prompt."""
        train_samples, val_samples, train_targets, val_targets = dataset_split

        best_prompt = prompt
        best_score = self.evaluator.evaluate(
            prompt,
            list(val_samples),
            list(val_targets),
        )
        patience_counter = 0

        for iteration in range(self.n_iterations):
            candidates = self._get_variants_from_best(
                best_prompt,
                n_candidates=self.n_candidates,
            )
            if not candidates:
                return best_prompt

            mini_batch_seed = (
                None if self.random_seed is None else self.random_seed + iteration
            )
            samples, sample_targets, _ = sample_mini_batch_with_indices(
                train_samples,
                train_targets,
                self.mini_batch_size,
                seed=mini_batch_seed,
            )
            if not samples:
                continue

            results: List[EvalResultDetailed] = [
                self.evaluator.evaluate(
                    candidate,
                    samples,
                    sample_targets,
                    failed_examples=self.k_samples,
                    return_detailed=True,
                )
                for candidate in candidates
            ]

            if sum(len(result.failed_examples) for result in results) == 0:
                resample_seed = (
                    None
                    if self.random_seed is None
                    else self.random_seed + 10_000 + iteration
                )
                samples, sample_targets, _ = sample_mini_batch_with_indices(
                    train_samples,
                    train_targets,
                    self.mini_batch_size,
                    seed=resample_seed,
                )
                results = [
                    self.evaluator.evaluate(
                        candidate,
                        samples,
                        sample_targets,
                        failed_examples=self.k_samples,
                        return_detailed=True,
                    )
                    for candidate in candidates
                ]

            selected = mmr_select(
                candidates=candidates,
                results=results,
                top_n=self.top_n_candidates,
                lambda_=_adaptive_lambda(best_score if best_score is not None else 0.0),
                bertscore_evaluate=_get_bertscore_evaluate(self.evaluator.metric),
            )
            if not selected:
                continue

            recommendations = self._generate_recommendations(
                selected,
                candidates,
                results,
                meta_info,
            )
            if recommendations:
                self.meta_prompt_module.update_section(
                    "recommendations",
                    recommendations,
                )

            score_before_iteration = best_score
            validation_score_cache = {best_prompt: best_score}

            for candidate_prompt, _ in selected:
                optimized_prompt = self.meta_prompt_module.optimize(
                    candidate_prompt,
                    meta_info=meta_info,
                )

                if optimized_prompt in validation_score_cache:
                    val_score = validation_score_cache[optimized_prompt]
                else:
                    val_score = self.evaluator.evaluate(
                        optimized_prompt,
                        list(val_samples),
                        list(val_targets),
                    )
                    validation_score_cache[optimized_prompt] = val_score

                if val_score is not None and best_score is not None and val_score > best_score:
                    best_score = val_score
                    best_prompt = optimized_prompt

            if best_score == score_before_iteration:
                patience_counter += 1
            else:
                patience_counter = 0

            if self.patience and patience_counter >= self.patience:
                break

        return best_prompt
