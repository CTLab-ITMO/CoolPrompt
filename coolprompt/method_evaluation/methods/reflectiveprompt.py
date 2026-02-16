from coolprompt.data_generator.generator import SyntheticDataGenerator
from coolprompt.method_evaluation.methods.autoprompting_method import (
    AutoPromptingMethod
)
from coolprompt.optimizer.reflective_prompt.run import reflectiveprompt


class ReflectivePromptMethod(AutoPromptingMethod):

    def _run(self, start_prompt: str) -> str:
        problem_description = self.config.get('problem_description')
        if problem_description is None:
            generator = SyntheticDataGenerator(self.model)
            problem_description = generator._generate_problem_description(
                prompt=start_prompt
            )

        final_prompt = reflectiveprompt(
            model=self.model,
            dataset_split=self.dataset_split,
            evaluator=self.evaluator,
            problem_description=problem_description,
            initial_prompt=start_prompt,
            population_size=self.config['method'].get('population_size', 10),
            num_epochs=self.config['method'].get('num_epochs', 5),
            output_path=self.config['method'].get(
                'output_path',
                "./reflectiveprompt_outputs"
            ),
            use_cache=self.config['method'].get('use_cache', False),
        )

        return final_prompt
