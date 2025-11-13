import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Degree abbreviation mapping
degree_map = {
    r"\bB\.?S\.?\b": "Bachelor's'",
    r"\bB\.?Sc\.?\b": "Bachelor's'",
    r"\bB\.?A\.?\b": "Bachelor of Arts",
    r"\bM\.?S\.?\b": "Master's'",
    r"\bMSc\b": "Master's'",
    r"\bM\.?A\.?\b": "Master's'",
    r"\bMBA\b": "Master of Business Administration",
    r"\bPh\.?D\.?\b": "Doctor of Philosophy",
    r"\bDBA\b": "Doctor of Business Administration",
    r"\bAssoc\.?\b": "Associate Degree",
    r"\bAI\b": "Artificial Intelligence",
    r"\bML\b": "Machine Learning",
    r"\bNLP\b": "Natural Language Processing",
    r"\bCS\b": "Computer Science",
    r"\bE\.?E\.?\b": "Electrical Engineering",
    r"\bComp\.?Sci\.?\b": "Computer Science",
    r"\bEng\.?\b": "Engineering",
    r"\bDL\b": "Deep Learning",
}

def expand_degree_abbreviations(text: str) -> str:
    """Expand degree abbreviations to full forms"""
    new_text = text
    for pattern, full in degree_map.items():
        new_text = re.sub(pattern, full, new_text, flags=re.IGNORECASE)
    return new_text 


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
        # Return the percentage of similar words between job_degree and candidate_degree
        cw= " ".join(candidate_degree).lower().split()
        jw= " ".join(job_degree).lower().split()
        matched_words = set(cw).intersection(set(jw))
        return len(matched_words) / max(len(set(jw)), 1)


def experience_match_score(job_experience, candidate_experience):
    """
    Calculates the experience match score between job experience and candidate experience.
    Returns a float between 0 and 1.
    """
    
    if not job_experience :
        return 1.0
    if not candidate_experience:
        return 0.5
    if candidate_experience >= job_experience:
        return 1.0
    else:
        return candidate_experience / job_experience

def total_match_score(job, candidate):
    """
    Calculates the total match score between job and candidate.
    Weights: skills 50%, degree 30%, experience 20%
    """
    skill_score = skill_match_score(job.get("Skills", []), candidate.get("Skills", []))
    degree_score = degree_match_score(job.get("Degree", []), candidate.get("Degree", []))
    cey=candidate.get("ExperianceYears", 0)
    cey=cey[0] if isinstance(cey, list) and len(cey) > 0 else cey
    jey=job.get("ExperianceYears", 0)
    jey=jey[0] if isinstance(jey, list) and len(jey) > 0 else jey
    experience_score = experience_match_score(jey, cey)

    total_score = (0.5 * skill_score) + (0.25 * degree_score) + (0.25 * experience_score)
    return total_score

if __name__ == "__main__":
    # Example usage
    print(degree_match_score(['Bachelor', 'Engineering', 'Management', 'Science'],["Bachelor'S'. In Electronics & Communications Engineering"]))