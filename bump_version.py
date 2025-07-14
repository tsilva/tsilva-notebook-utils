#!/usr/bin/env python3
"""
Simple script to bump version using bump2version
Usage: python bump_version.py [patch|minor|major]
"""
import subprocess
import sys

def bump_version(version_type="patch"):
    """Bump version using bump2version"""
    try:
        # Install bump2version if not available
        subprocess.run([sys.executable, "-m", "pip", "install", "bump2version"], 
                      capture_output=True, check=True)
        
        # Bump the version
        result = subprocess.run(["bump2version", version_type], 
                               capture_output=True, text=True, check=True)
        
        print(f"✅ Successfully bumped {version_type} version!")
        print("Changes made:")
        print(result.stdout)
        
        # Show the new version
        with open("pyproject.toml", "r") as f:
            for line in f:
                if line.startswith("version = "):
                    new_version = line.split('"')[1]
                    print(f"🏷️  New version: {new_version}")
                    break
                    
    except subprocess.CalledProcessError as e:
        print(f"❌ Error bumping version: {e}")
        print(f"stderr: {e.stderr}")
        sys.exit(1)

if __name__ == "__main__":
    version_type = sys.argv[1] if len(sys.argv) > 1 else "patch"
    
    if version_type not in ["patch", "minor", "major"]:
        print("Usage: python bump_version.py [patch|minor|major]")
        sys.exit(1)
        
    bump_version(version_type)
