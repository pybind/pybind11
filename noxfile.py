import nox


nox.options.sessions = ["lint", "tests", "tests_packaging"]


@nox.session(reuse_venv=True)
def lint(session: nox.Session) -> None:
    """
    Lint the codebase (except for clang-format/tidy).
    """
    session.install("pre-commit")
    session.run("pre-commit", "run", "-a")


@nox.session
def tests(session: nox.Session) -> None:
    """
    Run the tests (requires a compiler).
    """
    tmpdir = session.create_tmp()
    session.install("pytest", "cmake")
    session.run(
        "cmake",
        "-S",
        ".",
        "-B",
        tmpdir,
        "-DPYBIND11_WERROR=ON",
        "-DDOWNLOAD_CATCH=ON",
        "-DDOWNLOAD_EIGEN=ON",
        *session.posargs
    )
    session.run("cmake", "--build", tmpdir)
    session.run("cmake", "--build", tmpdir, "--config=Release", "--target", "check")


@nox.session
def tests_packaging(session: nox.Session) -> None:
    """
    Run the packaging tests.
    """

    session.install("-r", "tests/requirements.txt", "--prefer-binary")
    session.run("pytest", "tests/extra_python_package")


@nox.session(reuse_venv=True)
def docs(session: nox.Session) -> None:
    """
    Build the docs. Pass "serve" to serve.
    """

    session.install("-r", "docs/requirements.txt")
    session.chdir("docs")

    if "pdf" in session.posargs:
        session.run("sphinx-build", "-M", "latexpdf", ".", "_build")
        return

    session.run("sphinx-build", "-M", "html", ".", "_build")

    if "serve" in session.posargs:
        session.log("Launching docs at http://localhost:8000/ - use Ctrl-C to quit")
        session.run("python", "-m", "http.server", "8000", "-d", "_build/html")
    elif session.posargs:
        session.error("Unsupported argument to docs")


@nox.session(reuse_venv=True)
def make_changelog(session: nox.Session) -> None:
    """
    Inspect the closed issues and make entries for a changelog.
    """
    session.install("ghapi", "rich")
    session.run("python", "tools/make_changelog.py")


@nox.session(reuse_venv=True)
def build(session: nox.Session) -> None:
    """
    Build SDists and wheels.
    """

    session.install("build")
    session.run("python", "-m", "build")
    session.run("python", "-m", "build", env={"PYBIND11_GLOBAL_SDIST": "1"})
