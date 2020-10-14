# IMPORTANT: Should stay in sync with setup_helpers.py (mostly checked by CI /
# pre-commit).

from typing import Any, Iterator, Optional, Type, TypeVar, Union
from types import TracebackType

from distutils.command.build_ext import build_ext as _build_ext  # type: ignore
from distutils.extension import Extension as _Extension
import distutils.ccompiler
import contextlib

WIN: bool
PY2: bool
MACOS: bool
STD_TMPL: str

class Pybind11Extension(_Extension):
    def _add_cflags(self, *flags: str) -> None: ...
    def _add_lflags(self, *flags: str) -> None: ...
    def __init__(
        self, *args: Any, cxx_std: int = 0, language: str = "c++", **kwargs: Any
    ) -> None: ...
    @property
    def cxx_std(self) -> int: ...
    @cxx_std.setter
    def cxx_std(self, level: int) -> None: ...

@contextlib.contextmanager
def tmp_chdir() -> Iterator[str]: ...
def has_flag(compiler: distutils.ccompiler.CCompiler, flag: str) -> bool: ...
def auto_cpp_level(compiler: distutils.ccompiler.CCompiler) -> Union[int, str]: ...

class build_ext(_build_ext):  # type: ignore
    def build_extensions(self) -> None: ...

T = TypeVar("T", bound="ParallelCompile")

class ParallelCompile:
    def __init__(
        self, envvar: Optional[str] = None, default: int = 0, max: int = 0
    ): ...
    def function(self) -> Any: ...
    def install(self: T) -> T: ...
    def __enter__(self: T) -> T: ...
    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None: ...
