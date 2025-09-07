> Last updated: July 2025

# Context Generation for MCP and fast-agent

This repository holds a Python-based tool for generating context files, specifically for building AI agents with the Model Context Protocol (MCP) and the `fast-agent` framework.

> **Note:** These files are generated from source repositories and are time-sensitive. They are primarily for my own use, but feel free to explore the generation scripts.

## Overview

The core of this project is the `generate_context.py` script, which packages files from specified source repositories into comprehensive markdown files. The process is controlled by a series of job definitions in `config_context.py`, allowing for precise selection of files to include in each context package.

## How to Run

To generate the context files, run the main script from your terminal:

```sh
python generate_context.py --root-dir /path/to/your/repos --output-here
```

### Key Arguments:
*   `--root-dir`: **(Required)** The directory containing the source repos you want to process.
*   `--output-here` (`-oh`): A flag to save the output to a `generated_context/` folder in the current directory.
*   `--max-tokens`: Sets the token limit for including a single file (default: 16000). Files exceeding this are skipped.

## Configuration

The script's behavior is defined in `config_context.py`:

*   **`PACKAGING_JOBS`**: A list of dictionaries, where each one defines a single packaging task. You can specify the source repository, sub-path, include/ignore patterns, and the final output filename.
*   **`DEFAULT_IGNORE_PATTERNS`**: A set of file/folder patterns (like `.git`, `__pycache__`, `node_modules`) that are excluded from all jobs by default.

To customize the output, simply add or modify the jobs in the `PACKAGING_JOBS` list and rerun the script.
