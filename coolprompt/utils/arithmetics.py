import re


def clip(x, left, right):
    if x < left:
        return left
    if x > right:
        return right
    return x


def mean(lst):
    return sum(lst) / len(lst)


def extract_number_from_text(text):
    matches = re.findall(r'-?\d+(?:\.\d+)?', text)
    return matches[-1] if matches else None
