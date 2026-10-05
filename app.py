"""Root entry point for Hugging Face Spaces (and `streamlit run app.py`).

Hugging Face Spaces (Streamlit SDK) runs ``app.py`` by default. This shim simply
executes the real application in ``app/streamlit_app.py``.
"""
import os
import runpy

_APP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app", "streamlit_app.py")
runpy.run_path(_APP, run_name="__main__")
