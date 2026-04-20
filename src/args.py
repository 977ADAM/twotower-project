import argparse

class Args:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='A simple command-line tool.')
        self.parser.add_argument('--input', type=str, help='Input file path')
        self.parser.add_argument('--output', type=str, help='Output file path')
        self.parser.add_argument('--verbose', action='store_true', help='Enable verbose mode')

    def parse(self):
        return self.parser.parse_args()