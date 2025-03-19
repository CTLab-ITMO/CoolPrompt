import nltk
from nltk import TreebankWordDetokenizer


def detokenize(tokens):
    return TreebankWordDetokenizer().detokenize(tokens)


def traverse_tree(parsed_tree):
    phrases = []
    for tree in parsed_tree:
        if tree.label() == '_': continue
        phrases.append(detokenize(tree.leaves()))
        for subtree in tree:
            if type(subtree) == nltk.tree.Tree:
                if subtree.label() == '_': continue
                phrases.append(detokenize(subtree.leaves()))
                phrases.extend(traverse_tree(subtree))
    return phrases

def check_child(tree):
    check = False
    count = 0
    total_count = 0
    for subtree in tree:
        total_count += 1
        if type(subtree) == nltk.tree.Tree:
            if subtree.label() == '_':
                count += 1
    if count >= total_count - count: check = True

    return check

def collect_leaves(parsed_tree):
    leaves = []
    for tree in parsed_tree:
        if type(parsed_tree) != nltk.tree.Tree: continue
        if tree.label() == '_':
            leaves.append(detokenize(tree.leaves()))
            continue
        if check_child(tree): leaves.append(detokenize(tree.leaves()))
        else:
            leaves.extend(collect_leaves(tree))
    return leaves