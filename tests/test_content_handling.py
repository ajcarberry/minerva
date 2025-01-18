import pytest
from local_agent import LlamaAgent

@pytest.fixture
def test_agent(test_dir):
    """Create a test agent with a temporary directory structure"""
    return LlamaAgent(str(test_dir))

def test_extract_content():
    """Test content extraction from responses"""
    agent = LlamaAgent(".")
    
    # Test with marker
    response = "Here are some suggestions.\n---CONTENT---\n# Real Content"
    extracted = agent.extract_content(response)
    assert extracted == "# Real Content"
    
    # Test without marker
    response = "# Just Content"
    extracted = agent.extract_content(response)
    assert extracted == "# Just Content"

def test_load_all_documents(test_agent, tmp_path):
    """Test loading documents from directory"""
    # Create test files
    doc1_path = tmp_path / "docs" / "test1.md"
    doc2_path = tmp_path / "docs" / "test2.md"
    doc1_path.write_text("# Test 1")
    doc2_path.write_text("# Test 2")
    
    # Load documents
    context = test_agent.load_all_documents()
    
    # Verify context
    assert len(context) == 2
    assert any(doc["file"] == "test1.md" for doc in context)
    assert any(doc["content"] == "# Test 1" for doc in context)
