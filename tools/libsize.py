from __future__ import print_function
import os
import sys

# Internal build script for generating debugging test .so size.
# Usage:
#     python libsize.py file.so save.txt -- displays the size of file.so and, if save.txt exists, compares it to the
#                                           size in it, then overwrites save.txt with the new size for future runs.

if len(sys.argv) != 3:
    sys.exit("Invalid arguments: usage: python libsize.py file.so save.txt")

lib = sys.argv[1]
save = sys.argv[2]

libsize = -1

if not os.path.exists(lib):
    sys.exit("Error: requested file ({}) does not exist".format(lib))

libsize = os.path.getsize(lib)

print("------", os.path.basename(lib), "file size:", libsize, end='')

if os.path.exists(save):
    sf = open(save, 'r')
    oldsize = int(sf.readline())
    sf.close()

    if oldsize > 0:
        change = libsize - oldsize
        pct = change / oldsize * 100
        if change == 0:
            print(" (no change)", end='')
        elif change > 0:
            print(" (increase of {} bytes = {:.2f}%)".format(change, pct), end='')
        else:
            print(" (decrease of {} bytes = {:.2f}%)".format(-change, -pct), end='')

print()

sf = open(save, 'w')
sf.write(str(libsize))
sf.close()

