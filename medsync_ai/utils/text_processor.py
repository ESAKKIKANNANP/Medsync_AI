"""
Text processing utilities
"""
import re
from typing import List
from medsync_ai.config.settings import CHUNK_SIZE, CHUNK_OVERLAP

def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: Raw text to clean
    
    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep medical terms
    text = re.sub(r'[^\w\s\-.]', '', text)
    return text.strip()

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, 
               overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Text to chunk
        chunk_size: Size of each chunk in characters
        overlap: Overlap between chunks
    
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    
    return chunks

def extract_medical_terms(text: str) -> List[str]:
    """
    Extract medical terminology from text.
    
    Args:
        text: Medical text
    
    Returns:
        List of medical terms/entities
    """
    # Simple pattern matching - can be enhanced with NER models
    medical_patterns = [
        r'\b[A-Z][a-z]+\b(?:\s+[A-Z][a-z]+)*',  # Medical terms
        r'\b(?:ICD-|CPT-)\d+',  # Medical codes
    ]
    
    terms = []
    for pattern in medical_patterns:
        matches = re.findall(pattern, text)
        terms.extend(matches)
    
    return list(set(terms))
