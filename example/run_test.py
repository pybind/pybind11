import sys
import os
import re
import subprocess

remove_unicode_marker = re.compile(r'u(\'[^\']*\')')
remove_long_marker    = re.compile(r'([0-9])L')
remove_hex            = re.compile(r'0x[0-9a-fA-F]+')
shorten_floats        = re.compile(r'([1-9][0-9]*\.[0-9]{4})[0-9]*')


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
        line = line.replace('method of builtins.PyCapsule instance', '')
        line = line.strip()
        if sys.platform == 'win32':
            lower = line.lower()
            if 'constructor' in lower or 'destructor' in lower or 'ref' in lower:
                line = ""
        lines[i] = line

    lines = '\n'.join(sorted([l for l in lines if l != ""]))

    print('==================')
    print(lines)
    return lines

path = os.path.dirname(__file__)
if path != '':
    os.chdir(path)

name = sys.argv[1]
output_bytes = subprocess.check_output([sys.executable, name + ".py"])
output    = sanitize(output_bytes.decode('utf-8'))
reference = sanitize(open(name + '.ref', 'r').read())

if output == reference:
    print('Test "%s" succeeded.' % name)
    exit(0)
else:
    print('Test "%s" FAILED!' % name)
    exit(-1)
