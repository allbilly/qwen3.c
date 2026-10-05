"""Embed an OpenCL source verbatim in a C string for the experimental runner."""
from pathlib import Path
import json
import sys
source,output,name=sys.argv[1:]
Path(output).write_text(f'static const char {name}[] = '+json.dumps(Path(source).read_text())+';\n')
