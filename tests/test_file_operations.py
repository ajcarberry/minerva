import pytest
import os
import shutil
from pathlib import Path
from local_agent import LlamaAgent

@pytest.fixture
def test_agent(test_dir):
    """Create a test agent with a temporary directory structure"""
    return LlamaAgent(str(test_dir))

def test_create_new_file(test_agent):
    """Test creating a new markdown file"""
    # Test basic file creation
    assert test_agent.create_new_file("test_doc") == True
    assert os.path.exists(os.path.join(test_agent.project_path, "docs", "test_doc.md"))
    
    # Test file content
    with open(os.path.join(test_agent.project_path, "docs", "test_doc.md"), 'r') as f:
        content = f.read()
        assert content.startswith("# Test_Doc")
    
    # Test duplicate file creation
    assert test_agent.create_new_file("test_doc") == False

def test_get_staged_path(test_agent):
    """Test getting staged file path"""
    original_path = os.path.join(test_agent.project_path, "docs", "test.md")
    staged_path = test_agent.get_staged_path(original_path)
    assert staged_path.endswith(".staged_test.md")

def test_save_staged_changes(test_agent):
    """Test saving staged changes"""
    # Create original file
    test_agent.create_new_file("test_doc")
    original_path = os.path.join(test_agent.project_path, "docs", "test_doc.md")
    
    # Save staged changes
    new_content = "# Updated Content"
    test_agent.save_staged_changes(original_path, new_content)
    
    # Verify staged file exists
    staged_path = test_agent.get_staged_path(original_path)
    assert os.path.exists(staged_path)
    
    # Verify staged content
    with open(staged_path, 'r') as f:
        content = f.read()
        assert content == new_content