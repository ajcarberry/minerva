#!/usr/bin/env python3
import os
from .agent import LlamaAgent

def main():
    project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    agent = LlamaAgent(project_path)
    agent.start()

if __name__ == "__main__":
    main()