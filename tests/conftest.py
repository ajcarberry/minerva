import pytest
import os
import shutil
from pathlib import Path

@pytest.fixture
def test_dir(tmp_path):
    """Create a clean test directory structure for each test"""
    docs_dir = tmp_path / "docs"
    llama_dir = tmp_path / ".llama"
    
    # Create directories if they don't exist
    docs_dir.mkdir(exist_ok=True)
    llama_dir.mkdir(exist_ok=True)
    
    yield tmp_path
    
    # Cleanup
    if docs_dir.exists():
        shutil.rmtree(docs_dir)
    if llama_dir.exists():
        shutil.rmtree(llama_dir)