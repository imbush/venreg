#!/usr/bin/env python
"""Launch the venreg web app.  Usage:  python run.py [--port 8000]"""
import argparse
from venreg.server import serve

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    serve(ap.parse_args().port)
