from coolprompt.utils.correction.rule import Rule


def correct(prompt: str, rule: Rule, max_attempts: int = 3, **kwargs) -> str:
    """Running a correction loop. The provided prompt will be checked
    according to the `rule` and, if need to, fixed. Loop will end if the
    `prompt` is correct or after `max_attempts` attempts.

    Args:
        prompt (str): prompt to check.
        rule (Rule): rule which will be checked.
        max_attempts (optional, int): number of attempts the loop will end
            after. Defaults to 3.
        kwargs: other data explicit for the rule (e.g. start prompt, tag,
            etc.).
    Returns:
        result (str): corrected final prompt.
    """

    for _ in range(max_attempts):
        ok, meta = rule.check(prompt, **kwargs)

        if ok:
            return prompt

        prompt = rule.fix(prompt, meta)

        if rule.is_guaranteed_after_first_fix:
            break

    return prompt
