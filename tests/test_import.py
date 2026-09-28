def test_package_exposes_version() -> None:
    import math_modeling_agent

    assert math_modeling_agent.__version__ == "0.1.0"
