#!/usr/bin/env python3
"""
A script to package repository contents into a single context file.
It can operate in two modes:
1.  **Config-based Mode:** If configured repository directories are found, it runs
    pre-defined jobs from `config_context.py`.
2.  **General Mode:** If no configured repositories are found, it packages the
    entire directory where the script is run, respecting default ignore patterns.
"""
import os
import fnmatch
import argparse
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any, Set

# Attempt to import tiktoken for accurate token counting, but fall back gracefully.
try:
    import tiktoken
    TOKEN_COUNTING_AVAILABLE = True
except ImportError:
    TOKEN_COUNTING_AVAILABLE = False

# --- Configuration Import ---
from config_context import PACKAGING_JOBS, DEFAULT_IGNORE_PATTERNS, OUTPUT_DIR


class Project:
    """
    Represents a project to be packaged, handling file discovery, filtering, and writing.
    """
    def __init__(self, source_path: Path, output_path: Path, include_patterns: Optional[List[str]] = None, ignore_patterns: Optional[Set[str]] = None, max_tokens: int = 16000):
        self.source_path = source_path
        self.output_path = output_path
        self.include_patterns = include_patterns
        self.ignore_patterns = ignore_patterns if ignore_patterns is not None else DEFAULT_IGNORE_PATTERNS
        self.max_tokens = max_tokens
        self.stats = {"tokens": 0, "files": 0, "skipped_large": 0}

    def _count_tokens(self, text: str) -> int:
        """Counts tokens using tiktoken if available, otherwise falls back to word count."""
        if not text: return 0
        if TOKEN_COUNTING_AVAILABLE:
            try:
                encoding = tiktoken.get_encoding("cl100k_base")
                return len(encoding.encode(text, disallowed_special=()))
            except Exception:
                return len(text.split())
        return len(text.split())

    def _is_path_match(self, path: Path) -> bool:
        """Determines if a file's path matches the include/ignore patterns."""
        try:
            relative_path_str = path.relative_to(self.source_path).as_posix()
        except ValueError:
            # This can happen if the path is not inside the source_path, which might occur with symlinks.
            # In such cases, we treat it as a non-match.
            return False

        if any(fnmatch.fnmatch(relative_path_str, pattern) or fnmatch.fnmatch(path.name, pattern) for pattern in self.ignore_patterns):
            return False

        if not self.include_patterns:
            return True

        return any(fnmatch.fnmatch(relative_path_str, pattern) for pattern in self.include_patterns)

    def _build_directory_tree(self, files: List[Path]) -> str:
        """Builds a string representation of the directory structure for the included files."""
        tree = {}
        for file in files:
            # Ensure parts are relative to the source path for correct tree structure.
            try:
                parts = file.relative_to(self.source_path).parts
                node = tree
                for part in parts:
                    node = node.setdefault(part, {})
            except ValueError:
                continue
        
        def generate_tree_lines(d, prefix=""):
            lines = []
            # Sort items to ensure directories appear before files, then alphabetically.
            items = sorted(d.keys(), key=lambda k: (not bool(d[k]), k))
            for i, name in enumerate(items):
                connector = "└── " if i == len(items) - 1 else "├── "
                lines.append(f"{prefix}{connector}{name}")
                if d[name]:
                    extension = "    " if i == len(items) - 1 else "│   "
                    lines.extend(generate_tree_lines(d[name], prefix + extension))
            return lines

        tree_header = f"{self.source_path.name}/\n"
        return tree_header + "\n".join(generate_tree_lines(tree))

    def package(self) -> bool:
        """Main method to generate the context file. Returns True on success."""
        candidate_files = [path for path in self.source_path.rglob('*') if path.is_file() and self._is_path_match(path)]
        
        included_files = []
        for file_path in candidate_files:
            try:
                relative_path_str = file_path.relative_to(self.source_path).as_posix()
            except ValueError:
                continue # Skip files not relative to the source path

            # An explicit include bypasses the token check.
            is_explicitly_included = self.include_patterns and relative_path_str in self.include_patterns

            if not is_explicitly_included:
                try:
                    content = file_path.read_text(encoding='utf-8', errors='replace')
                    token_count = self._count_tokens(content)
                    if token_count > self.max_tokens:
                        print(f"  -> ⚠️  Skipping large file: {relative_path_str} ({token_count:,} tokens)")
                        self.stats["skipped_large"] += 1
                        continue
                except (OSError, UnicodeDecodeError):
                    # Skip files that can't be read for token counting.
                    print(f"  -> ⚠️  Skipping unreadable file: {relative_path_str}")
                    continue
            
            included_files.append(file_path)

        if not included_files:
            print("  -> ℹ️ No files matched the criteria for this job. Skipping.")
            return False

        self.output_path.parent.mkdir(exist_ok=True)
        with self.output_path.open("w", encoding="utf-8", errors="replace") as f:
            f.write(f"# Context for: {self.source_path.name}\n\n")
            f.write("## Directory Structure\n\n```\n")
            f.write(self._build_directory_tree(included_files))
            f.write("\n```\n---\n\n## File Contents\n\n")

            for file_path in sorted(included_files):
                try:
                    relative_path = file_path.relative_to(self.source_path).as_posix()
                    f.write(f"--- START OF FILE {relative_path} ---\n")
                    content = file_path.read_text(encoding='utf-8', errors='replace')
                    f.write(content.strip() + "\n")
                    self.stats["tokens"] += self._count_tokens(content)
                    self.stats["files"] += 1
                except (OSError, UnicodeDecodeError) as e:
                    f.write(f"[Error reading file: {e}]\n")
                except ValueError:
                    pass # Already handled
                f.write(f"--- END OF FILE {relative_path} ---\n\n\n")
        return True

    def get_stats(self) -> Dict[str, int]:
        return self.stats

def run_job(job: Dict[str, Any], root_dir: Path, output_dir: Path, max_tokens: int) -> Optional[Dict[str, int]]:
    """Processes a single packaging job."""
    print(f"▶️ Processing job: {job['output_filename']}")
    source_path = root_dir / job["repo_name"] / job.get("sub_path", ".")
    
    if not source_path.exists():
        print(f"  -> ⏭️ Skipping: Source path not found at '{source_path}'")
        return None

    # Combine default and job-specific ignore patterns.
    job_ignore_patterns = set(job.get("ignore", []))
    combined_ignore = DEFAULT_IGNORE_PATTERNS.union(job_ignore_patterns)

    project = Project(
        source_path=source_path,
        output_path=output_dir / job["output_filename"],
        include_patterns=job.get("include"),
        ignore_patterns=combined_ignore,
        max_tokens=max_tokens
    )
    
    if project.package():
        stats = project.get_stats()
        token_type = "tokens" if TOKEN_COUNTING_AVAILABLE else "words"
        skipped_info = f", {stats['skipped_large']} large files skipped" if stats['skipped_large'] > 0 else ""
        print(f"  -> ✅ Success! Packaged {stats['files']} files ({stats['tokens']:,} {token_type}{skipped_info}).")
        return stats
    return None

def run_general_mode(root_dir: Path, output_dir: Path, max_tokens: int) -> Optional[Dict[str, int]]:
    """Runs the script in general mode on the entire root directory."""
    print("▶️ No configured repos found. Running in general mode on the current directory.")
    
    # Use the root directory's name for the output file.
    output_filename = f"{root_dir.name}_context.md"
    output_path = output_dir / output_filename

    project = Project(
        source_path=root_dir,
        output_path=output_path,
        include_patterns=None,  # Include all files by default
        ignore_patterns=DEFAULT_IGNORE_PATTERNS, # Use the default ignore list
        max_tokens=max_tokens
    )

    if project.package():
        stats = project.get_stats()
        token_type = "tokens" if TOKEN_COUNTING_AVAILABLE else "words"
        skipped_info = f", {stats['skipped_large']} large files skipped" if stats['skipped_large'] > 0 else ""
        print(f"  -> ✅ Success! Packaged {stats['files']} files ({stats['tokens']:,} {token_type}{skipped_info}).")
        return stats
    return None

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Generate context files from source repositories based on `config_context.py`.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--root-dir", required=True, help="The root directory to process.")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help=f"Directory to save the generated files (default: '{OUTPUT_DIR}')")
    parser.add_argument("--output-here", "-oh", action="store_true", help="If set, output to 'generated_context' in the script's directory.")
    parser.add_argument("--max-tokens", type=int, default=16000, help="Maximum tokens for a single file to be included (default: 16000).")
    args = parser.parse_args()

    root_dir = Path(args.root_dir).resolve()
    output_dir = Path(args.output_dir)
    if args.output_here:
        output_dir = Path(__file__).parent / "generated_context"

    if not root_dir.is_dir():
        print(f"❌ Error: Root directory not found at '{root_dir}'")
        sys.exit(1)

    print("🚀 Starting context generation...")
    if not TOKEN_COUNTING_AVAILABLE:
        print("   (Note: `tiktoken` not found. Using word count for token stats.)")
    print("-" * 40)

    # --- Mode Detection ---
    configured_repos_found = any((root_dir / job["repo_name"]).exists() for job in PACKAGING_JOBS)

    total_stats = {"tokens": 0, "files": 0, "skipped_large": 0, "jobs": 0, "skipped_jobs": 0}

    if configured_repos_found:
        print("ℹ️ Configured repositories found. Running in config-based mode.")
        for job in PACKAGING_JOBS:
            stats = run_job(job, root_dir, output_dir, args.max_tokens)
            if stats:
                for key in ["tokens", "files", "skipped_large"]:
                    total_stats[key] += stats[key]
                total_stats["jobs"] += 1
            else:
                total_stats["skipped_jobs"] += 1
    else:
        stats = run_general_mode(root_dir, output_dir, args.max_tokens)
        if stats:
            for key in ["tokens", "files", "skipped_large"]:
                total_stats[key] += stats[key]
            total_stats["jobs"] += 1
        else:
            total_stats["skipped_jobs"] += 1
        
    print("-" * 40)
    print("\n🎉 All jobs complete!")
    job_plural = "job" if total_stats['jobs'] == 1 else "jobs"
    skipped_plural = "job" if total_stats['skipped_jobs'] == 1 else "jobs"
    print(f"  Summary: {total_stats['jobs']} {job_plural} successful, {total_stats['skipped_jobs']} {skipped_plural} skipped.")
    if total_stats["files"] > 0:
        token_type = "tokens" if TOKEN_COUNTING_AVAILABLE else "words"
        print(f"  📊 Total output: {total_stats['files']:,} files and {total_stats['tokens']:,} {token_type}.")
    if total_stats["skipped_large"] > 0:
        print(f"  ⏭️  Total large files skipped: {total_stats['skipped_large']:,}")

if __name__ == "__main__":
    main()
