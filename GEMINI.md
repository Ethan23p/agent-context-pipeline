# Project: Context Generation for MCP and fast-agent

## Project Overview

This project is a Python-based tool designed to generate context files for AI models, specifically tailored for the Model Context Protocol (MCP) and the `fast-agent` framework. The core functionality is encapsulated in a script that reads specified files from various source code repositories, and packages them into comprehensive markdown files. This process is highly configurable, allowing for precise control over the content included in each generated context file.

The primary technologies used are Python, with the optional `tiktoken` library for accurate token counting.

## Building and Running

There is no formal build process for this project. The context generation is handled by a single Python script.

**To run the context generation:**

```sh
python generate_context.py --root-dir /path/to/your/repos --output-here
```

**Key Command-Line Arguments:**

*   `--root-dir`: (Required) The path to the directory containing the source code repositories defined in the configuration.
*   `--output-dir`: The directory where the generated context files will be saved. Defaults to `generated_context`.
*   `--output-here` or `-oh`: A convenient flag to override the output directory and save the generated files to a `generated_context` folder within the project's root directory.
*   `--max-tokens`: Sets the maximum number of tokens for a single file to be included in the context. Defaults to 16000.

## Development Conventions

### Configuration

The context generation process is primarily driven by the `config_context.py` file. This file contains a list of `PACKAGING_JOBS`, where each job is a dictionary that defines:

*   `repo_name`: The name of the source repository.
*   `sub_path`: The subdirectory within the repository to process.
*   `output_filename`: The name of the markdown file to be generated.
*   `include`: A list of glob patterns specifying which files to include.
*   `ignore`: A list of glob patterns specifying which files to exclude.

### Ignore Patterns

A set of `DEFAULT_IGNORE_PATTERNS` is defined in `config_context.py` to exclude common files that are not useful for context, such as `.git` directories, `__pycache__`, virtual environments, and large media files. These defaults can be extended on a per-job basis.

### Token Counting

The script uses the `tiktoken` library if it is available to provide accurate token counts for the generated context files. If `tiktoken` is not installed, it falls back to a simple word count.

### Operating Modes

The `generate_context.py` script has two modes of operation:

1.  **Config-based Mode:** This is the primary mode, which runs when the script finds the configured repositories in the specified `--root-dir`. It processes each job defined in `PACKAGING_JOBS`.
2.  **General Mode:** If no configured repositories are found, the script will package the entire directory where the script is run, respecting the `DEFAULT_IGNORE_PATTERNS`.
