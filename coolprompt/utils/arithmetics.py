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
    extracted = re.findall(r'-?\d+(?:\.\d+)?', text)
    if len(extracted) == 0:
        return ""
    return extracted[-1]
