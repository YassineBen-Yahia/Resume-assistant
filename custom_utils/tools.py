from langchain_core.tools import tool
from pathlib import Path



@tool
def read_file_tool(file_path: str) -> str:
    """Tool to read the content of a file given its path."""
    path = Path(file_path)
    if not path.is_file():
        return f"Error: The file at {file_path} does not exist."
    from custom_utils.utils import extract_text_from_pdf
    return extract_text_from_pdf(file_path)

@tool
def get_degree_match_score(job_degree: list, candidate_degree: list) -> float:
    """Tool to get the degree match score between job and candidate degrees."""
    from custom_utils.fit_calc import degree_match_score
    from custom_utils.encoding_similarity import document_level_similarity

    score1 = document_level_similarity(job_degree, candidate_degree)
    score2 = degree_match_score(job_degree, candidate_degree)
    score= (score1 + score2) / 2
    return score

@tool
def get_skill_match_score(job_skills: list, candidate_skills: list) -> float:
    """Tool to get the skill match score between job and candidate skills."""
    from custom_utils.fit_calc import skill_match_score
    from custom_utils.encoding_similarity import skill_level_similarity

    score1= skill_match_score(job_skills, candidate_skills)
    score2 = skill_level_similarity(job_skills, candidate_skills)
    score = (score1 + score2.mean()) / 2

    return score

@tool
def get_experience_match_score(job_experience: int, candidate_experience: int) -> float:
    """Tool to get the experience match score between job and candidate experiences."""
    from custom_utils.fit_calc import experience_match_score

    return experience_match_score(job_experience, candidate_experience)

@tool
def get_total_match_score(job: dict, candidate: dict) -> float:
    """Tool to get the total match score between job and candidate."""




    from custom_utils.fit_calc import total_match_score
    
    exp = get_experience_match_score(job.get('ExperianceYears', 0), candidate.get('ExperianceYears', 0))
    skill = get_skill_match_score(job.get('Skills', []), candidate.get('Skills', []))
    degree = get_degree_match_score(job.get('Degree', []), candidate.get('Degree', []))
    score1 = 0.5 * skill + 0.2 * degree + 0.3 * exp
    score2 = total_match_score(job, candidate)
    score = (score1 + score2) / 2

    

    return score