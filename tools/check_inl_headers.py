#!/usr/bin/env python3
"""
Check the invariants of the -inl.h files used by the precompiled mode:

* Every namespace-scope function definition is marked PYBIND11_INLINE.
* Every macro in a preprocessor condition is known. A macro that changes the
  code must either be the same for the library and the modules (platform,
  compiler, Python version), or be encoded in PYBIND11_PRECOMPILED_CONFIG_CHECK
  (detail/internals.h) and listed in docs/compiling.rst.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Same in the library and in the modules that link it.
ENVIRONMENT_MACROS = {
    "GRAALPY_VERSION_NUM",
    "GRAALVM_PYTHON",
    "NDEBUG",
    "PYBIND11_BUILTIN_QUALNAME",
    "PYBIND11_HAS_CXXABI_H",
    "PYBIND11_HAS_STRING_VIEW",
    "PYBIND11_PRECOMPILED",
    "PYPY_VERSION",
    "PY_MAJOR_VERSION",
    "PY_MINOR_VERSION",
    "PY_VERSION_HEX",
    "Py_REF_DEBUG",
    "_MSC_VER",
    "_WIN32",
    "__GLIBCXX__",
    "__GNUC__",
    "__cpp_lib_unordered_map_try_emplace",
    "__clang__",
}

# Encoded in PYBIND11_PRECOMPILED_CONFIG_CHECK (PYBIND11_HAS_DIRECT_STRUCT_ACCESS through the
# Py_LIMITED_API bit and the environment).
GUARDED_MACROS = {
    "PYBIND11_BACKWARD_COMPATIBILITY_TP_DICTOFFSET",
    "PYBIND11_DETAILED_ERROR_MESSAGES",
    "PYBIND11_HAS_DIRECT_STRUCT_ACCESS",
    "PYBIND11_HAS_SUBINTERPRETER_SUPPORT",
    "PYBIND11_INTERNALS_VERSION",
    "PYBIND11_SIMPLE_GIL_MANAGEMENT",
    "Py_GIL_DISABLED",
    "Py_LIMITED_API",
}

# Only change code inside the library; documented in docs/compiling.rst.
LIBRARY_ONLY_MACROS = {
    "PYBIND11_DISABLE_NEW_STYLE_INIT_WARNING",
}

KNOWN_MACROS = ENVIRONMENT_MACROS | GUARDED_MACROS | LIBRARY_ONLY_MACROS

TOKENS = re.compile(
    r"""
    (?P<raw>R"(?P<delim>[^(\s]*)\(.*?\)(?P=delim)")
    | (?P<str>"(?:\\.|[^"\\\n])*")
    | (?P<chr>'(?:\\.|[^'\\\n])*')
    | (?P<line_comment>//[^\n]*)
    | (?P<block_comment>/\*.*?\*/)
    """,
    re.VERBOSE | re.DOTALL,
)
PP_CONDITION = re.compile(r"^[ \t]*#\s*(?:if|elif|ifdef|ifndef)\b(.*)$", re.MULTILINE)
PP_LINE = re.compile(r"^[ \t]*#.*$", re.MULTILINE)
NAMESPACE_MACRO = re.compile(
    r"^[ \t]*PYBIND11_(?:NAMESPACE_BEGIN|NAMESPACE_END|WARNING_\w+)\(.*\)[ \t]*$",
    re.MULTILINE,
)
IDENTIFIER = re.compile(r"\b[A-Za-z_]\w*\b")
NON_FUNCTION_BLOCK = re.compile(r"^(?:struct|class|union|enum)\b|=$")


def strip(text: str) -> str:
    """Blank out comments and string literals, keeping the line numbers."""

    def blank(match: re.Match[str]) -> str:
        if match.group("line_comment") or match.group("block_comment"):
            return re.sub(r"[^\n]", " ", match.group())
        return '""' + "\n" * match.group().count("\n")

    return TOKENS.sub(blank, text)


def check(path: Path) -> list[str]:
    text = strip(path.read_text(encoding="utf-8"))
    errors: list[str] = []

    for match in PP_CONDITION.finditer(text):
        line = text.count("\n", 0, match.start()) + 1
        errors.extend(
            f"{path}:{line}: unknown configuration macro {name}; see {Path(__file__).name}"
            for name in IDENTIFIER.findall(match.group(1))
            if name != "defined" and name not in KNOWN_MACROS
        )

    code = PP_LINE.sub(lambda m: " " * len(m.group()), text)
    code = NAMESPACE_MACRO.sub(lambda m: " " * len(m.group()), code)

    # One entry per open brace: True for namespace and extern "C" blocks, whose
    # contents are still at namespace scope.
    scopes: list[bool] = []
    start = 0
    for pos, char in enumerate(code):
        at_namespace_scope = all(scopes)
        if char == "{":
            if not at_namespace_scope:
                scopes.append(False)
                continue
            header = " ".join(code[start:pos].split())
            transparent = header == 'extern "C"' or header.startswith("namespace")
            if (
                not transparent
                and "(" in header
                and not NON_FUNCTION_BLOCK.search(header)
                and "PYBIND11_INLINE" not in header.split()
            ):
                offset = start + len(code[start:pos]) - len(code[start:pos].lstrip())
                line = code.count("\n", 0, offset) + 1
                errors.append(f"{path}:{line}: definition without PYBIND11_INLINE")
            scopes.append(transparent)
            if transparent:
                start = pos + 1
        elif char == "}":
            scopes.pop()
            if all(scopes):
                start = pos + 1
        elif char == ";" and at_namespace_scope:
            start = pos + 1

    return errors


def main(argv: list[str]) -> int:
    errors = [error for arg in argv for error in check(Path(arg))]
    for error in errors:
        print(error)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
