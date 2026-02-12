#!/usr/bin/env python3
"""
Interactive utility to find and prune files in your project that aren't in the template.

Run this after `copier copy/update` to clean up orphaned files that were
removed from the template but still exist in your project.

How it works:
1. Reads your .copier-answers.yml to get template source and answers
2. Generates a fresh temporary copy of the template with the same answers
3. Compares your project against the fresh copy
4. Shows files that exist in your project but not in the fresh template
5. Lets you interactively delete orphaned files
"""

import difflib
import filecmp
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


def load_copier_answers(project_root: Path) -> dict:
    """Load .copier-answers.yml to find template source and answers."""
    answers_file = project_root / ".copier-answers.yml"
    if not answers_file.exists():
        print(f"Error: {answers_file} not found. Are you in a copier-managed project?")
        sys.exit(1)

    with open(answers_file) as f:
        return yaml.safe_load(f)


def generate_fresh_template(template_src: str, answers: dict, temp_dir: Path) -> Path:
    """
    Generate a fresh copy of the template using copier with the same answers.

    Returns the path to the generated project.
    """
    # Resolve template path to absolute before changing directories
    template_path = Path(template_src)
    if not template_path.is_absolute():
        template_path = Path.cwd() / template_src
    template_path = template_path.resolve()

    if not template_path.exists():
        raise ValueError(f"Template path {template_path} does not exist")

    # Create a temporary answers file
    answers_file = temp_dir / "answers.yml"
    with open(answers_file, "w") as f:
        yaml.safe_dump(answers, f)

    output_dir = temp_dir / "output"
    output_dir.mkdir()

    # Change to temp directory so we can use relative paths
    original_cwd = Path.cwd()
    try:
        os.chdir(temp_dir)

        # Run copier copy with absolute template path and relative paths for temp files
        cmd = [
            "copier",
            "copy",
            "--force",
            "--answers-file",
            "answers.yml",  # Relative path
            str(template_path),  # Absolute path to template
            "output",  # Relative path
        ]

        # copier is a trusted local dependency here; inputs are controlled by the project.
        subprocess.run(  # noqa: S603
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        return output_dir
    except subprocess.CalledProcessError as e:
        print("Error: Failed to generate fresh template copy")
        print(f"Command: {' '.join(cmd)}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: 'copier' command not found. Please install copier:")
        print("  pip install copier")
        print("  # or")
        print("  uv pip install copier")
        sys.exit(1)
    finally:
        # Always restore original working directory
        os.chdir(original_cwd)


def get_file_set(root: Path) -> set[Path]:
    """
    Get all files in a directory (relative paths).

    Excludes common build/cache directories and copier metadata.
    """
    exclude_dirs = {
        ".git",
        "__pycache__",
        ".venv",
        "venv",
        ".tox",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "node_modules",
        "dist",
        "build",
        ".eggs",
        ".coverage",
        "htmlcov",
        "site",
    }

    files = set()

    for item in root.rglob("*"):
        if item.is_dir():
            continue

        rel_path = item.relative_to(root)
        parts = rel_path.parts

        # Skip excluded directories
        if any(part in exclude_dirs or part.endswith(".egg-info") for part in parts):
            continue

        files.add(rel_path)

    return files


def colorize(text: str, color: str) -> str:
    """Simple ANSI color wrapper."""
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "reset": "\033[0m",
    }
    return f"{colors.get(color, '')}{text}{colors['reset']}"


def get_modified_files(project_root: Path, template_root: Path) -> list[Path]:
    """Find files that exist in both roots but differ in content."""
    template_files = get_file_set(template_root)
    project_files = get_file_set(project_root)

    common_files = template_files & project_files
    modified = []

    for rel_path in sorted(common_files):
        template_file = template_root / rel_path
        project_file = project_root / rel_path
        if not filecmp.cmp(template_file, project_file, shallow=False):
            modified.append(rel_path)

    return modified


def print_unified_diff(template_root: Path, project_root: Path, rel_path: Path) -> None:
    """Print a unified diff for a text file; skip binary-like content."""
    template_file = template_root / rel_path
    project_file = project_root / rel_path

    try:
        template_text = template_file.read_text(encoding="utf-8")
        project_text = project_file.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        print(colorize(f"  (binary or non-utf8) {rel_path}", "yellow"))
        return

    diff_lines = difflib.unified_diff(
        template_text.splitlines(keepends=True),
        project_text.splitlines(keepends=True),
        fromfile=f"template/{rel_path}",
        tofile=f"project/{rel_path}",
    )

    diff_output = "".join(diff_lines)
    if diff_output:
        print(diff_output)


def interactive_prune(project_root: Path, orphaned_files: list[Path]) -> None:
    """Interactively prompt user to delete orphaned files."""
    if not orphaned_files:
        print(colorize("✓ No orphaned files found. Project is in sync with template.", "green"))
        return

    print(colorize(f"\nFound {len(orphaned_files)} file(s) not in template:", "yellow"))
    print()

    for f in sorted(orphaned_files):
        print(f"  {f}")

    print()
    print("Options:")
    print("  [a] Delete all")
    print("  [i] Interactive (prompt for each file)")
    print("  [n] Keep all (no changes)")

    choice = input("\nChoose action [a/i/n]: ").strip().lower()

    if choice == "n":
        print("No changes made.")
        return

    deleted_count = 0

    if choice == "a":
        for file_path in sorted(orphaned_files):
            full_path = project_root / file_path
            try:
                full_path.unlink()
                print(colorize(f"✓ Deleted: {file_path}", "red"))
                deleted_count += 1
            except Exception as e:
                print(colorize(f"✗ Failed to delete {file_path}: {e}", "red"))

    elif choice == "i":
        for file_path in sorted(orphaned_files):
            full_path = project_root / file_path
            response = input(f"Delete {file_path}? [y/N]: ").strip().lower()
            if response == "y":
                try:
                    full_path.unlink()
                    print(colorize("  ✓ Deleted", "red"))
                    deleted_count += 1
                except Exception as e:
                    print(colorize(f"  ✗ Failed: {e}", "red"))
            else:
                print(colorize("  → Kept", "green"))

    else:
        print("Invalid choice. No changes made.")
        return

    print()
    print(colorize(f"✓ Deleted {deleted_count} file(s).", "green"))

    # Clean up empty directories
    cleanup_empty_dirs(project_root)


def cleanup_empty_dirs(project_root: Path) -> None:
    """Remove empty directories after file deletion."""
    removed_dirs = []

    for dirpath in sorted(project_root.rglob("*"), reverse=True):
        if not dirpath.is_dir():
            continue

        # Skip excluded dirs
        if any(p in {".git", "__pycache__", ".venv", "venv"} for p in dirpath.parts):
            continue

        try:
            if not any(dirpath.iterdir()):
                dirpath.rmdir()
                removed_dirs.append(dirpath.relative_to(project_root))
        except OSError:
            pass

    if removed_dirs:
        print(colorize(f"✓ Removed {len(removed_dirs)} empty director(ies).", "green"))


def main():
    project_root = Path.cwd()

    print(colorize("Template Prune Utility", "blue"))
    print("=" * 50)

    # Load copier metadata
    answers = load_copier_answers(project_root)
    template_src = answers.get("_src_path")

    if not template_src:
        print("Error: Could not find _src_path in .copier-answers.yml")
        sys.exit(1)

    print(f"Template source: {template_src}")
    print("Generating fresh template copy for comparison...")

    # Generate a fresh template copy in a temp directory
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        try:
            fresh_copy = generate_fresh_template(template_src, answers, temp_dir)
        except Exception as e:
            print(f"Error generating fresh template: {e}")
            sys.exit(1)

        print(colorize("✓ Fresh template generated", "green"))
        print()

        # Compare files
        template_files = get_file_set(fresh_copy)
        project_files = get_file_set(project_root)

        # Find orphaned files (in project but not in fresh template)
        # Exclude .copier-answers.yml itself and common ignore patterns
        orphaned = project_files - template_files
        orphaned = {f for f in orphaned if f.name not in {".copier-answers.yml", ".git"}}

        modified = get_modified_files(project_root, fresh_copy)
        if modified:
            print(colorize(f"Found {len(modified)} modified file(s):", "yellow"))
            for rel_path in modified:
                print(f"  {rel_path}")

            show_diff = input("\nShow diffs for modified files? [y/N]: ").strip().lower()
            if show_diff == "y":
                print()
                for rel_path in modified:
                    print(colorize(f"--- Diff: {rel_path}", "blue"))
                    print_unified_diff(fresh_copy, project_root, rel_path)
                    print()

        # Interactive pruning
        interactive_prune(project_root, list(orphaned))


if __name__ == "__main__":
    main()
