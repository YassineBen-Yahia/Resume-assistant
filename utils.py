import re
import random
from pathlib import Path
import PyPDF2
import fitz  # PyMuPDF


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


def experience_advice(job_experience, candidate_experience):
    """
    Generates advice based on experience comparison.
    """
    if not job_experience:
        return "Experience is not a requirement for this job."
    if candidate_experience >= job_experience:
        return "You meet the experience requirement for this job."
    else:
        diff = job_experience - candidate_experience
        advice = f"You are {diff} years short of the required experience. Consider gaining more experience through relevant roles, projects, or further education."
        advice += " Highlight any transferable skills or related experiences you have that may compensate for the gap."
        return advice
    
def generate_general_advice(job_requirements, resume_data):
    """
    Generates general advice based on job requirements and resume data.
    """
    advice = ""
    
    # Skills advice
    missing = missing_skills(job_requirements.get('Skills', []), resume_data.get('Skills', []))
    if missing:
        advice+=generate_advice_for_missing_skills(missing)
    
    # Experience advice
    job_exp = job_requirements.get('ExperianceYears', ['0 years'])[0]
    resume_exp = resume_data.get('ExperianceYears', ['0 years'])[0]
    
    try:
        job_years = sum_experience_years(job_exp)
        if job_years==0:
            pass
        resume_years = sum_experience_years(resume_exp)
        advice+=experience_advice(job_years, resume_years)
    except:
        pass

    return advice



def generate_resume_summary(entities):
    """Generate a summary of the extracted resume information"""
    summary = []
    
    if 'Name' in entities:
        summary.append(f"👤 Name: {', '.join(entities['Name'][:2])}")
    
    if 'Skills' in entities:
        skill_count = len(entities['Skills'])
        summary.append(f"🛠️ Skills: {skill_count} identified")
    
    if 'ExperianceYears' in entities:
        summary.append(f"📅 Experience: {entities['ExperianceYears'][0] if entities['ExperianceYears'] else 'Not specified'}")
    
    if 'Degree' in entities:
        summary.append(f"🎓 Education: {len(entities['Degree'])} qualification(s)")
    
    if 'Designation' in entities:
        summary.append(f"💼 Current Role: {entities['Designation'][0] if entities['Designation'] else 'Not specified'}")
    
    return summary



def get_fit_grade(fit_score):
    """Convert fit score to letter grade"""
    
    if fit_score >= 0.7:
        return "Very good fit "
    elif fit_score >= 0.5:
        return "Good fit "
    elif fit_score >= 0.4:
        return "fair fit "

    elif fit_score < 0.4:
        return "no fit"


def generate_job_analysis(job_requirements, resume_data, fit_score):
    """Generate analysis of job match with detailed fit score breakdown"""
    analysis = {
        'fit_percentage': round(fit_score * 100, 1),
        'fit_grade': get_fit_grade(fit_score),
        'strengths': [],
        'gaps': [],
        'recommendations': []
    }
    
    # Analyze skills match
    missing_skills_list = missing_skills(job_requirements.get('Skills', []), resume_data.get('Skills', []))
    matched_skills = set(job_requirements.get('Skills', [])).intersection(set(resume_data.get('Skills', [])))
    
    if matched_skills:
        analysis['strengths'].append(f"✅ Matching skills ({len(matched_skills)}): {', '.join(list(matched_skills)[:5])}")
    
    if missing_skills_list:
        analysis['gaps'].append(f"❌ Missing skills ({len(missing_skills_list)}): {', '.join(list(missing_skills_list)[:5])}")
        analysis['recommendations'].append(f"🎯 Consider learning: {', '.join(list(missing_skills_list)[:3])}")
    
    # Analyze experience
    job_exp = job_requirements.get('ExperianceYears', ['0 years'])[0]
    resume_exp = resume_data.get('ExperianceYears', ['0 years'])[0]
    
    try:
        job_years = int(re.search(r'\d+', job_exp).group()) if re.search(r'\d+', job_exp) else 0
        resume_years = int(re.search(r'\d+', resume_exp).group()) if re.search(r'\d+', resume_exp) else 0
        
        if resume_years >= job_years:
            analysis['strengths'].append(f"✅ Experience requirement met: {resume_years} years (required: {job_years})")
        else:
            gap = job_years - resume_years
            analysis['gaps'].append(f"❌ Experience gap: Need {job_years} years, have {resume_years} years ({gap} year gap)")
    except:
        pass
    
    # Analyze education
    job_degrees = set(job_requirements.get('Degree', []))
    resume_degrees = set(resume_data.get('Degree', []))
    
    if job_degrees.intersection(resume_degrees):
        analysis['strengths'].append("✅ Education requirements met")
    elif job_degrees:
        analysis['gaps'].append("❌ Education requirements not fully met")
        analysis['recommendations'].append("📚 Consider relevant certifications or education")
    
    # Generate overall recommendation based on fit score
    if fit_score >= 0.8:
        analysis['recommendations'].insert(0, "🎉 Excellent match! You should definitely apply for this position.")
    elif fit_score >= 0.6:
        analysis['recommendations'].insert(0, "👍 Good match! Address the gaps mentioned above to strengthen your application.")
    elif fit_score >= 0.4:
        analysis['recommendations'].insert(0, "⚠️ Moderate match. Focus on developing missing skills before applying.")
    else:
        analysis['recommendations'].insert(0, "📈 Consider gaining more relevant experience and skills before applying.")
    
    return analysis


def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file"""
    try:
        # You can pass either a file path (str or Path) or a file-like object to fitz.open.
        # If pdf_file is a path, it works directly.
        # If pdf_file is a file-like object (e.g., from an upload), use fitz.open(stream=pdf_file.read(), filetype="pdf")
        if isinstance(pdf_file, (str, Path)):
            f = fitz.open(pdf_file)
        else:
            f = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = " "
        for page in f:
            text = text + str(page.get_text())
        text = text.strip()
        text = ' '.join(text.split())
        return text
    except Exception as e:
        return f"Error extracting text from PDF: {str(e)}"

def skills_mapping(skills_list):
    """
    Maps skills in skills_list to standardized skills using skills_dict.
    """
    new_skills_list = []

    skills_dict = {
        "python": ["python", "python3","data analysis"],
        "java": ["java"],
        "c++": ["c++", "cpp"],
        "javascript": ["javascript", "js"],
        "sql": ["sql", "mysql", "postgresql", "sqlite", "mongodb"],
        "html": ["html","web development", "web designing"],
        "css": ["css","web design", "web designing"],
        "react": ["react", "reactjs"],
        "node.js": ["node.js", "nodejs", "node"],
        "django": ["django"],
        "flask": ["flask"],
        "deep learning": ["deep learning", "dl","PyTorch", "tensorflow", "keras", "neural networks","nlp","natural language processing", "computer vision", "cv", "opencv"],
        "docker": ["docker", "containerization", "containers"],
        "kubernetes": ["kubernetes", "k8s"],
        "machine learning": ["machine learning", "ml"," artificial intelligence", "ai","xgboost","scikit-learn","sklearn"],
        "data analysis": ["data analysis", "data analytics","matplotlib","pandas","numpy","data science"],
        "project management": ["project management", "pm"],
        "agile methodologies": ["agile methodologies", "agile"],
        "communication": ["communication", "communicating","problem solving", "problem-solving"],
        "problem solving": ["problem solving", "problem-solving"],
        "leadership": ["leadership", "leading"],
        "time management": ["time management", "time-management"],
        "teamwork": ["teamwork", "team work"],
        "critical thinking": ["critical thinking", "critical-thinking"],
        "ai": ["artificial intelligence", "ai", "machine learning", "ml", "deep learning", "dl","neural networks","nlp","natural language processing", "computer vision", "cv", "opencv","pytorch", "tensorflow", "keras","artificial neural networks"],
        "nlp": ["natural language processing", "nlp","python","python3","pytorch","tensorflow","keras"],
        "cv": ["computer vision", "cv","python","python3","pytorch","tensorflow","keras","opencv"],
        "data science": ["data science", "data analytics", "data analysis","python","python3","pandas","numpy","matplotlib"],
        "full stack development": ["full stack development", "full-stack development","web development","web designing","html","css","javascript","js","react","node.js","nodejs","node","django","flask"],
        "web development": ["web development","web designing","html","css","javascript","js","react","node.js","nodejs","node","django","flask"],
        "web designing": ["web designing","web development","html","css","javascript","js","react","node.js","nodejs","node","django","flask"], 
        "data engineering": ["data engineering", "data engineer","sql","mysql","postgresql","sqlite","mongodb","python","python3","pandas","numpy"],
        "devops": ["devops","docker","kubernetes","k8s","ci/cd","continuous integration","continuous deployment"],
        "ci/cd": ["ci/cd","continuous integration","continuous deployment","devops","docker","kubernetes","k8s"],



    }
    for skill in skills_list:
        skill_lower = skill.lower()
        for standard_skill, variants in skills_dict.items():
            if skill_lower in standard_skill or skill_lower in variants:
                new_skills_list+= skills_dict[standard_skill]
            else:
                new_skills_list+= [skill]
                


    return list(set(new_skills_list))