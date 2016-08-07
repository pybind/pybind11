import sys
import os
import re
import subprocess
import difflib

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
        if line.startswith("### "):
            # Constructor/destructor output.  Useful for example, but unreliable across compilers;
            # testing of proper construction/destruction occurs with ConstructorStats mechanism instead
            line = ""
        line = remove_unicode_marker.sub(r'\1', line)
        line = remove_long_marker.sub(r'\1', line)
        line = remove_hex.sub(r'0', line)
        line = shorten_floats.sub(r'\1', line)
        line = line.replace('__builtin__', 'builtins')
        line = line.replace('example.', '')
        line = line.replace('unicode', 'str')
        line = line.replace('ExampleWithEnum.EMode', 'EMode')
        line = line.replace('example.EMode', 'EMode')
        line = line.replace('method of builtins.PyCapsule instance', '')
        line = line.strip()
        lines[i] = line

    return '\n'.join(sorted([l for l in lines if l != ""]))

path = os.path.dirname(__file__)
if path != '':
    os.chdir(path)

if len(sys.argv) < 2:
    print("Syntax: %s <test name>" % sys.argv[0])
    exit(0)

name = sys.argv[1]
try:
    output_bytes = subprocess.check_output([sys.executable, "-u", name + ".py"],
                                           stderr=subprocess.STDOUT)
except subprocess.CalledProcessError as e:
    if e.returncode == 99:
        print('Test "%s" could not be run.' % name)
        exit(0)
    else:
        raise

output    = sanitize(output_bytes.decode('utf-8'))
reference = sanitize(open(name + '.ref', 'r').read())

if output == reference:
    print('Test "%s" succeeded.' % name)
    exit(0)
else:
    print('Test "%s" FAILED!' % name)
    print('--- output')
    print('+++ reference')
    print(''.join(difflib.ndiff(output.splitlines(True),
                                reference.splitlines(True))))
    exit(-1)
