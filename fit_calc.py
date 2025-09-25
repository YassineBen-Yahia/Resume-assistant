import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils import expand_degree_abbreviations


def semantic_similarity(list1, list2):
    doc1 = ' '.join(list1).lower()
    doc2 = ' '.join(list2).lower()
    # remove punctuation
    doc1 = re.sub(r"[^\w\s]", " ", doc1)
    doc2 = re.sub(r"[^\w\s]", " ", doc2)

    vectorizer = TfidfVectorizer(ngram_range=(1,2))
    tfidf_matrix = vectorizer.fit_transform([doc1, doc2])
    if tfidf_matrix.shape[1] == 0:
        return 0.0
    return float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])


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
    # Normalize and expand abbreviations so that "BSc" and "Bachelor of Science" are treated the same
    for i in range(len(job_degree)):
        job_degree[i] = expand_degree_abbreviations(job_degree[i])
    for i in range(len(candidate_degree)):
        candidate_degree[i] = expand_degree_abbreviations(candidate_degree[i])
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
    job_degree = ["Bachelor's in Machine learning", "Master's in AI"]
    candidate_degree = ["Master's' in Artificial Intelligence", "BSc in Computer Science"]
    print("Degree Match Score:", degree_match_score(job_degree, candidate_degree))
    print (expand_degree_abbreviations("BSc. in Computer Science, M.A. in Economics, Ph.D. in Physics"))