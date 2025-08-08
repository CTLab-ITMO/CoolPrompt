# DistillPrompt

## LLM-Based AutoPrompting method

This method is based on different prompt-transformation methods that will use LLM to effectively examine the search area and find the best prompt variations for given task.

### Workflow

<p align="center">
    <picture>
    <source srcset="../../../docs/images/distillPrompt.jpg">
    <img alt="DistillPrompt workflow" width="100%" height="100%">
    </picture>
</p>

Every epoch consist of these steps:
- 4 (can be changed) new variations of the prompt are generated based on best previous solution.
- 5 random samples from train dataset are getting integrated into prompts to increase their quality.
- Compressing all new prompts, distilling them into a couple of sentences with the main ideas.
- Aggregation of prompts. It is useful to combine the new points of view that were created during this epoch.
- Paraphrasing of the aggregated candidate in order to explore the local optimum better.