import importlib


def test_package_imports() -> None:
    app = importlib.import_module("app")
    assert app.__name__ == "app"
