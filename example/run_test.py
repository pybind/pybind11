import sys
import os
import re
import subprocess

remove_unicode_marker = re.compile(r'u(\'[^\']*\')')
remove_long_marker    = re.compile(r'([0-9])L')
remove_hex            = re.compile(r'0x[0-9a-fA-F]+')
shorten_floats        = re.compile(r'([1-9][0-9]*\.[0-9]{4})[0-9]*')

relaxed = False

def sanitize(lines):
    lines = lines.split('\n')
    for i in range(len(lines)):
        line = lines[i]
        if line.startswith(" |"):
            line = ""
        line = remove_unicode_marker.sub(r'\1', line)
        line = remove_long_marker.sub(r'\1', line)
        line = remove_hex.sub(r'0', line)
        line = shorten_floats.sub(r'\1', line)
        line = line.replace('__builtin__', 'builtins')
        line = line.replace('example.', '')
        line = line.replace('unicode', 'str')
        line = line.replace('Example4.EMode', 'EMode')
        line = line.replace('example.EMode', 'EMode')
        line = line.replace('method of builtins.PyCapsule instance', '')
        line = line.strip()
        if relaxed:
            lower = line.lower()
            # The precise pattern of allocations and deallocations is dependent on the compiler
            # and optimization level, so we unfortunately can't reliably check it in this kind of test case
            if 'constructor' in lower or 'destructor' in lower \
               or 'ref' in lower or 'freeing' in lower:
                line = ""
        lines[i] = line

    lines = '\n'.join(sorted([l for l in lines if l != ""]))

    print('==================')
    print(lines)
    return lines

path = os.path.dirname(__file__)
if path != '':
    os.chdir(path)

if len(sys.argv) < 2:
    print("Syntax: %s [--relaxed] <test name>" % sys.argv[0])
    exit(0)

if len(sys.argv) == 3 and sys.argv[1] == '--relaxed':
    del sys.argv[1]
    relaxed = True

name = sys.argv[1]
output_bytes = subprocess.check_output([sys.executable, name + ".py"],
                                       stderr=subprocess.STDOUT)

output    = sanitize(output_bytes.decode('utf-8'))
reference = sanitize(open(name + '.ref', 'r').read())

if 'NumPy missing' in output:
    print('Test "%s" could not be run.' % name)
    exit(0)
elif output == reference:
    print('Test "%s" succeeded.' % name)
    exit(0)
else:
    print('Test "%s" FAILED!' % name)
    exit(-1)
