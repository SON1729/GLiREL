import importlib.util
import subprocess
import sys
from pathlib import Path

root=Path(__file__).resolve().parent
if importlib.util.find_spec('streamlit') is None:
    subprocess.run([sys.executable,'-m','pip','install','-r',str(root/'requirements.txt')],check=True)
subprocess.run([sys.executable,'-m','streamlit','run',str(root/'app.py'),
    '--server.address','127.0.0.1','--server.port','8502','--theme.base','light',
    '--browser.gatherUsageStats','false'],cwd=root,check=True)
