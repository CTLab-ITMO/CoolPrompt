# ReflectivePrompt

## Evolutionary-Based AutoPrompting method

This method is based on the idea of Reflective Evolution and text-based gradients. 

It implements short-term and long-term reflections to provide some clarifications and make crossover and mutation operations more precise and effective.

<b>IMPORTANT NOTE</b>: ReflectivePrompt is critical to the given <u>"problem description"</u> argument, so if you want to achieve a better score with it - it's important to provide detailed problem description to the assistant. But if you are not able to do so or just don't want to - no worries, problem description will be generated automatically!

#### Reflections
- **Short-term** - is based on 2 parent prompts and is unique for each couple of parents. LLM by knowing which of two prompts if better should provide some hints to improve them.
- **Long-term** - updates every epoch based on previous version of itself and current short-term reflections. Accumulates the best ideas and hints for prompt optimization during whole evolution process.

#### Operations
- **Crossover** - creates a new prompt based on two parents and their short-term reflection. Tries to combine better sides of each parent to create 'stronger' and more 'powerful' (in terms of fitness-function) individual.
- **Elitist mutation** - creates a batch of new prompts based on elitist-prompt from current population and current long-term reflection. Tries to create a bunch of prompts that are similar to elitist therefore explores the search space better.

### Workflow
<p align="center">
    <picture>
    <source srcset="../../../docs/images/reflectivePrompt.jpg">
    <img alt="ReflectivePrompt workflow" width="100%" height="100%">
    </picture>
</p>