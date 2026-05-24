import unittest

import pandas as pd


def _assert_contract(df):
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["input_data", "target"]
    assert len(df) > 0
    assert df["input_data"].map(type).eq(str).all()
    assert df["target"].map(type).eq(str).all()


class TestNewLoaders(unittest.TestCase):
    def test_ifeval(self):
        from src.utils.load_dataset_ifeval import load_ifeval
        _assert_contract(load_ifeval(max_rows=10))

    def test_gsm8k(self):
        from src.utils.load_dataset_math import load_gsm8k
        _assert_contract(load_gsm8k())

    def test_svamp(self):
        from src.utils.load_dataset_math import load_svamp
        _assert_contract(load_svamp())

    def test_sst2(self):
        from src.utils.load_dataset_classification import (
            load_sst2,
        )
        _assert_contract(load_sst2())

    def test_rucola(self):
        from src.utils.load_dataset_russian import load_rucola
        _assert_contract(load_rucola())


if __name__ == "__main__":
    unittest.main()
