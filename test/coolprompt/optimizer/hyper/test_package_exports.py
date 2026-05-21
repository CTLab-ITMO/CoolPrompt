def test_hyper_package_exports():
    import coolprompt.optimizer.hyper as hyper_pkg

    names = set(hyper_pkg.__all__)
    assert "MetaPromptOptimizer" in names
    assert "HyPERLightMethod" in names
    assert "HyPEROptimizer" in names
    assert "HyPERMethod" in names
    assert "Optimizer" in names
