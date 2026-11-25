import importlib.util
import subprocess
import sys
import os
import argparse
from pathlib import Path

root=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description='GLiREL visual workbench')
parser.add_argument('--local-model',action='store_true',help='Enable sentence inference on this machine')
parser.add_argument('--port',type=int,default=8502)
args=parser.parse_args()
if args.local_model:
    os.environ['GLIREL_LOCAL_INFERENCE']='1'
else:
    os.environ.pop('GLIREL_LOCAL_INFERENCE',None)
if importlib.util.find_spec('streamlit') is None:
    subprocess.run([sys.executable,'-m','pip','install','-r',str(root/'requirements.txt')],check=True)
subprocess.run([sys.executable,'-m','streamlit','run',str(root/'app.py'),
    '--server.address','127.0.0.1','--server.port',str(args.port),'--theme.base','light',
    '--browser.gatherUsageStats','false'],cwd=root,check=True)
