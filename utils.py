import re
import random
def standardize_data(parsed_data: dict) -> dict:
    def clean_text(text):
        text = text.strip()
        text = re.sub(r"\s+", " ", text)
        text = text.replace("’", "'")
        return text

    def normalize_skill(skill):
        skill = clean_text(skill).lower()
        skill = skill.replace("c + +", "c++").replace("c +", "c")
        skill = re.sub(r"\(.*?\)", "", skill)
        return skill.title()

    def normalize_company(name):
        name = clean_text(name)
        name = re.sub(r"\b(inc\.?|corp\.?|ltd\.?|llc)\b", "", name, flags=re.IGNORECASE)
        return name.strip().title()

    def normalize_experience(exp):
        match = re.search(r"(\d+)", exp)
        return int(match.group(1)) if match else None

    standardized = {}

    for key, values in parsed_data.items():
        if not isinstance(values, list):
            continue

        cleaned_values = []
        for v in values:
            v = clean_text(v)

            if key.lower() == "skills":
                v = normalize_skill(v)

            elif key.lower() in ["companies worked at", "org"]:
                v = normalize_company(v)

            elif key.lower() in ["experience", "experianceyears"]:
                num = normalize_experience(v)
                if num is not None:
                    cleaned_values.append(num)
                    continue

            else:
                v = v.title()

            cleaned_values.append(v)

        standardized[key] = list(dict.fromkeys(cleaned_values))

    return standardized


def sum_experience_years(experience_list):
    """
    Sums up the years in a list of experience strings like ['15 years', '10 year', '5 years'].
    Returns the total years as an integer.
    """
    total = 0
    for exp in experience_list:
        parts = exp.split()
        if parts and parts[0].isdigit():
            total += int(parts[0])
    return total

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
    r"\bCS\b": "Computer Science",
    r"\bE\.?E\.?\b": "Electrical Engineering",
    r"\bComp\.?Sci\.?\b": "Computer Science",
    r"\bEng\.?\b": "Engineering",
    r"\bDL\b": "Deep Learning",
}


def expand_degree_abbreviations(text: str) -> str:
    new_text = text
    for pattern, full in degree_map.items():
        new_text = re.sub(pattern, full, new_text, flags=re.IGNORECASE)
    return new_text


def missing_skills(job_skills, candidate_skills):
    """
    Returns a list of skills that are in job_skills but not in candidate_skills.
    """
    
    if not job_skills:
        return []
    if not candidate_skills:
        return job_skills
    missing = set(job_skills) - set(candidate_skills)
    return list(missing)        

def generate_advice_for_missing_skills(missing_skills):
    """
    Generates advice for missing skills.
    """
    if not missing_skills:
        return "You have all the required skills for this job."
    #missing_skills.shuffle()
    advice = "To improve your chances for this job, consider acquiring the following skills: "
    advice += ", ".join(missing_skills)
    advice += ". You can take online courses, attend workshops, or gain practical experience in these areas."
    advice += " Additionally, highlight any related skills or experiences you have that may be relevant."
    i= random.random()
    if i>0.5:
        advice += " Consider working on personal projects to demonstrate your skills then add them to your portfolio or resume."
    return advice

