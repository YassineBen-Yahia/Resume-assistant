import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def semantic_similarity(list1, list2):
    """
    Calculates the semantic similarity between two lists of strings using TF-IDF and cosine similarity.
    Returns a float between 0 and 1.
    """
    # Combine lists into documents
    doc1 = ' '.join(list1)
    doc2 = ' '.join(list2)
    # Vectorize
    vectorizer = TfidfVectorizer().fit([doc1, doc2])
    tfidf_matrix = vectorizer.transform([doc1, doc2])
    # Compute cosine similarity
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return similarity


def skill_match_score(job_skills, candidate_skills):
    """
    Calculates the skill match score between job skills and candidate skills.
    Returns a float between 0 and 1.
    """
    if not job_skills or not candidate_skills:
        return 0.0
    matched_skills = set(job_skills).intersection(set(candidate_skills))
    score = len(matched_skills) / len(set(job_skills))
    return score*0.9 + 0.1 * semantic_similarity(job_skills, candidate_skills)


def degree_match_score(job_degree, candidate_degree):
    """
    Calculates the degree match score between job degree and candidate degree.
    Returns 1.0 if they match, else 0.0.
    """
    matched = set(job_degree).intersection(set(candidate_degree))
    semantic_similarity_score = semantic_similarity(job_degree, candidate_degree)
    if matched:
        return 1.0
    elif semantic_similarity_score > 0.5:
        return 0.5 + 0.5 * semantic_similarity_score
    else:
        return semantic_similarity_score 

if __name__ == "__main__":
    # Example usage
    job_degree = ["Bachelor's in Computer Science", "Master's in Data Science"]
    candidate_degree = ["BSc in Computer Engineering", "MSc in Data Analytics"]
    print("Degree Match Score:", degree_match_score(job_degree, candidate_degree))