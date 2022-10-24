from pathlib import Path

DIR = Path(__file__).parent.resolve()


def test_build_import(virtualenv):
    virtualenv.run('pip install "pybind11==2.8.0" --no-build-isolation')
    virtualenv.env["EXAMPLE_NAME"] = "pet"
    virtualenv.run(f"pip install {DIR} --no-build-isolation")

    virtualenv.run(f"pip install {DIR.parent.parent} --no-build-isolation")
    virtualenv.env["EXAMPLE_NAME"] = "dog"
    virtualenv.run(f"pip install {DIR} --no-build-isolation")

    script = DIR / "check_installed.py"
    virtualenv.run(f"python {script}")
