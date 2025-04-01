#!/usr/bin/env python3

from dotenv import load_dotenv
load_dotenv()

import os
import re
import subprocess
import sys
from enum import Enum
import argparse

class BumpType(Enum):
    PATCH = "patch"
    MINOR = "minor"
    MAJOR = "major"


def get_current_version():
    """Extract the current version from setup.py."""
    with open("setup.py", "r") as f:
        content = f.read()

    match = re.search(r'version="(\d+\.\d+\.\d+)"', content)
    if not match:
        raise ValueError("Could not find version string in setup.py")

    return match.group(1)


def bump_version(version, bump_type):
    """Bump the version according to semver rules."""
    major, minor, patch = map(int, version.split("."))

    if bump_type == BumpType.PATCH:
        patch += 1
    elif bump_type == BumpType.MINOR:
        minor += 1
        patch = 0
    elif bump_type == BumpType.MAJOR:
        major += 1
        minor = 0
        patch = 0

    return f"{major}.{minor}.{patch}"


def update_version_in_setup(new_version):
    """Update the version in setup.py."""
    with open("setup.py", "r") as f:
        content = f.read()

    updated_content = re.sub(
        r'version="(\d+\.\d+\.\d+)"',
        f'version="{new_version}"',
        content
    )

    with open("setup.py", "w") as f:
        f.write(updated_content)


def run_command(command, description):
    """Run a shell command and handle errors."""
    print(f"\n{description}...\n")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"Error: {description} failed with exit code {result.returncode}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Publish package to PyPI")
    parser.add_argument(
        "--bump",
        type=str,
        choices=[t.value for t in BumpType],
        default=BumpType.PATCH.value,
        help="Version bump type (patch, minor, major)"
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Build the package but don't upload to PyPI"
    )
    args = parser.parse_args()

    # Check for uncommitted changes
    result = subprocess.run(
        "git diff-index --quiet HEAD --",
        shell=True
    )
    if result.returncode != 0:
        print("Error: There are uncommitted changes. Commit or stash them first.")
        sys.exit(1)

    # Get current version and bump it
    current_version = get_current_version()
    new_version = bump_version(current_version, BumpType(args.bump))
    print(f"Bumping version: {current_version} -> {new_version}")

    # Update version in setup.py
    update_version_in_setup(new_version)

    # Build the package
    run_command("rm -rf dist/ build/ *.egg-info", "Cleaning build directories")
    run_command("python setup.py sdist bdist_wheel", "Building package")

    # Upload to PyPI if not disabled
    if not args.no_upload:
        # Load credentials from .env
        os.environ["TWINE_USERNAME"] = "__token__"
        os.environ["TWINE_PASSWORD"] = os.getenv("PYPI_API_TOKEN")

        run_command("twine check dist/*", "Checking package")
        run_command("twine upload dist/*", "Uploading to PyPI")

    # Commit and tag the new version
    run_command(f'git add setup.py', "Staging changes")
    run_command(f'git commit -m "Bump version to {new_version}"', "Committing changes")
    run_command(f'git tag v{new_version}', "Tagging release")

    print(f"\nVersion {new_version} published successfully!")
    print("\nDon't forget to push changes and tags:")
    print("  git push && git push --tags")


if __name__ == "__main__":
    main()
