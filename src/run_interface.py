#!/usr/bin/env python3
"""
Launcher script for the CANCapital Gradio Interface

This runs the gradio interface using pixi to ensure all dependencies are available.
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Launch the Gradio interface using pixi"""

    # Get the current directory
    project_dir = Path(__file__).parent

    print("🚀 Launching CANCapital EDA & Modeling Interface...")
    print(f"📁 Project directory: {project_dir}")
    print("🐍 Using pixi environment for all dependencies")

    try:
        # Run the gradio interface using pixi
        _ = subprocess.run(
            ["pixi", "run", "python", str(project_dir / "gradio_interface.py")],
            check=True,
        )

    except subprocess.CalledProcessError as e:
        print(f"❌ Error running interface: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Interface stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

        sys.exit(1)

if __name__ == "__main__":
    main()
