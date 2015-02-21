#!/bin/env python3

import os
import unittest


def load_tests(loader, tests, pattern):
    base_dir = os.path.dirname(__file__)
    pattern = pattern or 'test.*'
    packages = loader.discover(start_dir=base_dir, pattern=pattern)
    tests.addTests(packages)
    return tests


if __name__ == '__main__':
    unittest.main()
