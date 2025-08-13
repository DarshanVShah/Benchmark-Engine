"""
Command-line interface for the BenchmarkEngine framework.

This module provides the CLI entry point and argument parsing functionality.
"""

import argparse
import os
import sys


def main():
    """CLI entry point for the BenchmarkEngine framework."""
    parser = argparse.ArgumentParser(description="BenchmarkEngine Framework")
    parser.add_argument("--example", action="store_true", help="Run basic example")
    parser.add_argument(
        "--custom", action="store_true", help="Run custom plugins example"
    )

    args = parser.parse_args()

    if args.example:
        # Import and run basic example
        sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
        from examples.dummy_example import main as run_basic

        run_basic()
    elif args.custom:
        # Import and run custom example
        sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
        from examples.hf_example import main as run_custom

        run_custom()
    else:
        print("BenchmarkEngine Framework")
        print("=" * 30)
        print("Use --example to run the basic benchmark")
        print("Use --custom to run the custom plugins example")
        print("\nFor more information, see the README.md file")


if __name__ == "__main__":
    main()
