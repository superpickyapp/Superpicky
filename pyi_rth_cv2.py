# -*- coding: utf-8 -*-
"""
PyInstaller runtime hook to prevent cv2 binary extension recursion error.

This hook ensures cv2 is loaded before any other module tries to import it,
preventing the "recursion is detected during loading of cv2 binary extensions" error.
"""

import os
import sys

# Prevent cv2 loader from causing recursion issues
# by setting the cv2 module path explicitly
def _setup_cv2():
    """Pre-load cv2 to avoid recursion during lazy loading."""
    try:
        # Force cv2 to be imported early and properly
        import cv2
        # Print version for debugging (will appear in console if run from terminal)
        # print(f"cv2 version: {cv2.__version__}")
    except ImportError as e:
        print(f"Warning: Could not import cv2: {e}")
    except Exception as e:
        print(f"Warning: cv2 import error: {e}")

# Run setup
_setup_cv2()
