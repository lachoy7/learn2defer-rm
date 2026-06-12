#!/usr/bin/env python3
import _bootstrap  # noqa: F401

from eval.analysis import _parse_args, main

if __name__ == "__main__":
    results_path = _parse_args()
    main(results_path)
