from __future__ import print_function

import argparse
import sys
import sysconfig

from . import get_include


def print_includes():
    dirs = [sysconfig.get_path('include')]
    if sysconfig.get_path('platinclude') not in dirs:
        dirs.append(sysconfig.get_path('platinclude'))
    if get_include() not in dirs:
        dirs.append(get_include())
    print(' '.join('-I' + d for d in dirs))


def main():
    parser = argparse.ArgumentParser(prog='python -m pybind11')
    parser.add_argument('--includes', action='store_true',
                        help='Include flags for both pybind11 and Python headers.')
    args = parser.parse_args()
    if not sys.argv[1:]:
        parser.print_help()
    if args.includes:
        print_includes()


if __name__ == '__main__':
    main()
