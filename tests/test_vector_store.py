"""
Tests for the vector store implementation.
"""
import os
import pytest
from local_agent.vector_store import VectorStore

def test_vector_store_initialization(tmp_path):
    """Test that the vector store initializes correctly."""
    # Create a vector store using a temporary directory
    store = VectorStore(str(tmp_path))
    
    # Verify the persist directory was created
    vector_store_path = os.path.join(tmp_path, '.llama', 'vector_store')
    assert os.path.exists(vector_store_path)
    
    # Verify we can get the collection
    collection = store.get_collection()
    assert collection is not None
    assert collection.name == "documents"

def test_vector_store_reset(tmp_path):
    """Test that the vector store can be reset."""
    store = VectorStore(str(tmp_path))
    
    # Get initial collection
    initial_collection = store.get_collection()
    
    # Reset the store
    store.reset()
    
    # Get new collection
    new_collection = store.get_collection()
    
    # Verify we have a new collection
    assert new_collection is not None
    assert new_collection.name == "documents"
    # Note: ChromaDB maintains the same object reference even after reset
    assert new_collection == initial_collection
