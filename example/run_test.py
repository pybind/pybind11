import subprocess, sys, os

path = os.path.dirname(__file__)
if path != '':
    os.chdir(path)

name = sys.argv[1]
output    = subprocess.check_output([sys.executable, name + ".py"]).decode('utf-8')
reference = open(name + '.ref', 'r').read()

if output == reference:
    print('Test "%s" succeeded.' % name)
    exit(0)
else:
    print('Test "%s" FAILED!' % name)
    exit(-1)
