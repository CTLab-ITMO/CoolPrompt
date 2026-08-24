def test_rider_package_exports_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    import coolprompt.optimizer.rider as rider_pkg

    assert "RIDEROptimizer" in rider_pkg.__all__
    assert "RIDERGenesisMethod" in rider_pkg.__all__
    assert rider_pkg.RIDERGenesisMethod().name == "rider"
