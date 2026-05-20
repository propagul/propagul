#!/usr/bin/env python3
"""Script to download and update local JS assets (Air-Gap Resilience).

This script fetches the pinned versions of external libraries
and saves them to the propagul/server/assets directory.
"""

import os
import urllib.request
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Define target directory relative to the repository root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(REPO_ROOT, "python", "propagul", "server", "assets")

ASSETS = {
    "htmx.min.js": "https://unpkg.com/htmx.org@2.0.4/dist/htmx.min.js",
    "chart.umd.js": "https://unpkg.com/chart.js@4.4.1/dist/chart.umd.js"
}

def main():
    if not os.path.exists(ASSETS_DIR):
        os.makedirs(ASSETS_DIR)
        logging.info(f"Created assets directory at {ASSETS_DIR}")

    for filename, url in ASSETS.items():
        filepath = os.path.join(ASSETS_DIR, filename)
        logging.info(f"Downloading {filename} from {url} ...")
        try:
            urllib.request.urlretrieve(url, filepath)
            file_size = os.path.getsize(filepath)
            logging.info(f"Successfully saved {filename} ({file_size / 1024:.1f} KB)")
        except Exception as e:
            logging.error(f"Failed to download {filename}: {e}")

if __name__ == "__main__":
    main()
