from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from typing import List, Dict

from custom_utils.utils import extract_text_from_pdf
from custom_utils.Process_data import process
from custom_utils.fit_calc import (
    degree_match_score,
    skill_match_score,
    experience_match_score,
    total_match_score
)
from custom_utils.encoding_similarity import (
    document_level_similarity,
    skill_level_similarity
)
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification
import spacy
from pathlib import Path



path=Path(r"model-best")
# Load the best trained model
nlp = spacy.load(path)

model_name = "manishiitg/resume-ner"  # This is trained for NER
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

model_name2 = "Shrav20/job-ner-deberta"  # This is trained for NER
tokenizer2 = AutoTokenizer.from_pretrained(model_name)
model2 = AutoModelForTokenClassification.from_pretrained(model_name)


def _get_degree_match_score(job_degree: List[str], candidate_degree: List[str]) -> float:
    score1 = document_level_similarity(job_degree, candidate_degree)
    score2 = degree_match_score(job_degree, candidate_degree)
    return (score1 + score2) / 2


def _get_skill_match_score(job_skills: List[str], candidate_skills: List[str]) -> float:
    score1 = skill_match_score(job_skills, candidate_skills)
    score3 = document_level_similarity(job_skills, candidate_skills)
    score2 = skill_level_similarity(job_skills, candidate_skills)
    return (score1 + score2.mean() + score3) / 3


def _get_experience_match_score(job_experience: int, candidate_experience: int) -> float:
    return experience_match_score(job_experience, candidate_experience)


def _get_total_match_score(job: Dict, candidate: Dict) -> float:
    print(candidate)
    exp = _get_experience_match_score(
        job.get("ExperianceYears", 0),
        candidate.get("ExperianceYears", 0),
    )
    skill = _get_skill_match_score(
        job.get("Skills", []),
        candidate.get("Skills", []),
    )
    degree = _get_degree_match_score(
        job.get("Degree", []),
        candidate.get("Degree", []),
    )

    score1 = 0.5 * skill + 0.2 * degree + 0.3 * exp
    score2 = total_match_score(job, candidate)

    return (score1 * 0.7) + (score2 * 0.3)




class ReadFileInput(BaseModel):
    file_path: str = Field(..., description="Absolute path to PDF resume file")

def read_file_tool_func(file_path: str) -> str:
    text = extract_text_from_pdf(file_path)
    return {"context": text} 

read_file_tool = StructuredTool(
    name="read_file_tool",
    description="Read content from a PDF resume file at a given path.",
    func=read_file_tool_func,
    args_schema=ReadFileInput,
)


class MatchScoreInput(BaseModel):
    job_description: str = Field(..., description="Job description details")
    candidate_resume: str = Field(..., description="Path to the candidate's resume PDF file")

def match_score_tool_func(job_description: str, candidate_resume: str) -> float:
    resume_text = read_file_tool_func(candidate_resume)["context"]
    C, J = process(nlp, model, tokenizer, model2,tokenizer2, resume_text, job_description)
    return _get_total_match_score(J, C)

match_score_tool = StructuredTool(
    name="get_total_match_score",
    description="Read the resume from the file and compute total match score combining skills, degrees and experience.",
    func=match_score_tool_func,
    args_schema=MatchScoreInput,
)



TOOLS = [read_file_tool, match_score_tool]