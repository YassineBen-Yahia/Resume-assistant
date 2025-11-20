from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Mean Pooling function
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Encode function
def encode(texts):
    """Encode texts into embeddings"""
    if isinstance(texts, str):
        texts = [texts]
    
    # Tokenize
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    
    # Compute embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
    
    # Perform pooling
    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    
    # Normalize embeddings
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    
    return embeddings.numpy()

# Example job skills lists

# Method 1: Encode individual skills and compare
def skill_level_similarity(skills1, skills2):
    """Compare skills at individual level"""
    # Encode all skills
    all_skills = skills1 + skills2
    embeddings = encode(all_skills)
    
    # Split embeddings
    emb1 = embeddings[:len(skills1)]
    emb2 = embeddings[len(skills1):]
    
    # Calculate pairwise similarities
    similarity_matrix = cosine_similarity(emb1, emb2)
    
    print("Skill-level similarity matrix:")
    print(similarity_matrix)
    print(f"\nAverage similarity: {similarity_matrix.mean():.4f}")
    print(f"Max similarity: {similarity_matrix.max():.4f}")
    
    return similarity_matrix

# Method 2: Encode entire skill lists as documents
def document_level_similarity(skills1, skills2):
    """Compare entire skill lists as documents"""
    # Join skills into text documents
    doc1 = ", ".join(skills1)
    doc2 = ", ".join(skills2)
    
    # Encode documents
    embeddings = encode([doc1, doc2])
    
    # Calculate similarity
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    
    print(f"Document-level similarity: {similarity:.4f}")
    
    return similarity





if __name__ == "__main__":
    job1_skills = [
        "Python programming",
        "Machine learning",
        "Data analysis",
        "SQL databases",
        "Communication skills"
    ]

    job2_skills = [
        "Writing",
        "Arts",
        "Statistical analysis",
        "Sculpting",
        "Team collaboration"
    ]
    job2_skills=['Bachelor', 'Engineering', 'Electronics', 'Science']
    job1_skills=["Bachelor'S'. In Electronics & Communications Engineering"]

    # Run all methods
    print("=" * 50)
    print("Job 1 Skills:", job1_skills)
    print("Job 2 Skills:", job2_skills)
    print("=" * 50)

    print("\n--- Method 1: Skill-level Comparison ---")
    p1=skill_level_similarity(job1_skills, job2_skills)

    print("\n--- Method 2: Document-level Comparison ---")
    p2=document_level_similarity(job1_skills, job2_skills)

    print("\nSummary of Similarities:")
    print((p1.mean()+p2)/2)  # Simple average of both methods



