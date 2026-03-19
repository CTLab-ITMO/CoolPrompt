import yaml

from coolprompt.method_evaluation.methods import (
    ReflectivePromptMethod
)


def evaluate_method(
    method: str,
    config: dict | str,
    start_prompt: str,
    output_file_path: str = "./method_evaluation_output.yaml",
    saving_model_answers: bool = False
) -> None:
    """Evaluating autoprompting method.
    Stores the results into output yaml file.
        Path to file can be specified.

    Args:
        method (str): Name of the method to evaluate.
            Supported methods: ['reflectiveprompt'].
        config (dict | str): Either provided config
            or string path to config yaml file.
        start_prompt (str): start prompt.
        output_file_path (str): Filepath to save the results.
            Defaults to "./method_evaluation_output.yaml".
        saving_model_answers (bool):
            Either to save all model answers on test subset or not.
            If True - the path to save-file can be provided through config
                ("model_answers_output_path" parameter)
                Defaults to "./model_answers.yaml".

    Returns:
        Tuple[List[str], List[str]]: loaded dataset and targets
    """

    if isinstance(config, str):
        config_file_path = config
        with open(config_file_path, 'r') as file:
            config = yaml.safe_load(file)

    match method:
        case "reflectiveprompt":
            autoprompting_method = ReflectivePromptMethod(config)
        case _:
            raise ValueError(f"Unsupported method name: {method}")

    autoprompting_method.run(
        start_prompt
    )

    with open(output_file_path, 'w') as file:
        yaml.safe_dump(
            {
                'dataset': config['dataset']['name'],
                'configuration': config['dataset']['configuration'],
                'start_prompt': start_prompt,
                'final_prompt': autoprompting_method.final_prompt,
                'val_score': autoprompting_method.final_val_score,
                'test_score': autoprompting_method.final_test_score
            },
            file
        )
