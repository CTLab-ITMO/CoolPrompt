import pytest


@pytest.fixture(autouse=True)
def reset_hyper_bertscore_singleton():
    """Avoid cross-test pollution of ``hyper._bertscore_evaluate``."""
    import coolprompt.optimizer.hyper.hyper as hyper_mod

    hyper_mod._bertscore_evaluate = None
    yield
    hyper_mod._bertscore_evaluate = None
