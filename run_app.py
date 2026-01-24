import os
import sys
import streamlit.web.cli as stcli

def resolve_path(path):
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, path)
    return os.path.join(os.getcwd(), path)

if __name__ == "__main__":
    try:
        # Configure Streamlit environment variables
        os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
        os.environ["STREAMLIT_SERVER_HEADLESS"] = "false"
        
        print("Starting Reconciliation App (Patched v7)...", flush=True)
        
        # Monkeypatch Streamlit credentials to bypass email prompt
        # Use dynamic imports to avoid PyInstaller analysis crash and handle missing modules gracefully
        try:
            print(f"Patching credentials module...", flush=True)
            import importlib
            
            # Dynamically import credentials
            credentials = importlib.import_module("streamlit.runtime.credentials")
            _Activation = credentials._Activation
            
            # Patch email_prompt to return empty string
            credentials.email_prompt = lambda: ""
            
            # Patch _check_activated
            def _mock_check_activated(self, auto_resolve=True):
                if self.activation is None:
                    self.activation = _Activation(email="", is_valid=True)

            credentials.Credentials._check_activated = _mock_check_activated
            credentials.Credentials.check_activated = _mock_check_activated
            print("Credentials patched successfully.", flush=True)
            
            # Also patch bootstrap if possible
            try:
                bootstrap = importlib.import_module("streamlit.web.bootstrap")
                bootstrap.email_prompt = lambda: ""
                print("Patched bootstrap.email_prompt", flush=True)
            except ImportError:
                pass
                
        except Exception as e:
            print(f"Warning: Failed to patch credentials: {e}", flush=True)

        app_path = resolve_path("app.py")
        print(f"App path resolved to: {app_path}", flush=True)
        
        if not os.path.exists(app_path):
            print(f"ERROR: File not found at {app_path}", flush=True)
            input("Press Enter to exit...")
            sys.exit(1)

        # Set sys.argv for Streamlit
        sys.argv = [
            "streamlit",
            "run",
            app_path,
            "--global.developmentMode=false",
            "--server.headless=false",
            "--server.maxUploadSize=1000",
            "--browser.gatherUsageStats=false",
            "--server.address=localhost",
        ]
        
        print("Launching Streamlit...", flush=True)
        sys.exit(stcli.main())
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"An error occurred: {e}", flush=True)
        input("Press Enter to exit...")
