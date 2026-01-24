import PyInstaller.__main__
import streamlit
import os

# Get Streamlit path for datas
streamlit_path = os.path.dirname(streamlit.__file__)

print(f"Streamlit found at: {streamlit_path}")

# Define PyInstaller args
args = [
    'run_app.py',  # Entry point
    '--name=ReconciliationAppV6',
    '--onefile',
    '--clean',
    '--noconfirm',
    # Include Streamlit static files (Critical for UI)
    f'--add-data={streamlit_path}/static;streamlit/static',
    f'--add-data={streamlit_path}/runtime;streamlit/runtime',
    # Include our app files
    '--add-data=app.py;.',
    '--add-data=reconciliation_core.py;.',
    '--add-data=.streamlit/config.toml;.streamlit',
    # Hidden imports needed by Streamlit/Pandas
    '--hidden-import=streamlit',
    '--hidden-import=pandas',
    '--hidden-import=openpyxl',
    '--hidden-import=streamlit.runtime.scriptrunner.magic_funcs',
    '--hidden-import=streamlit.runtime.scriptrunner.script_runner',
    # Exclude unnecessary heavy modules to keep size down
    '--exclude-module=matplotlib',
    '--exclude-module=scipy',
    '--exclude-module=hook-streamlit', # Avoid recursion
]

print("🚀 Building executable... This may take 2-3 minutes.")
try:
    PyInstaller.__main__.run(args)
    print("✅ Build complete! Check the 'dist' folder.")
except Exception as e:
    print(f"❌ Build failed: {e}")
