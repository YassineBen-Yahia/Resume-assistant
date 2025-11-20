EMAIL_PATTERN = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
SKILLS_PATTERN=r"\b(Python|Java|C\+\+|C#|JavaScript|TypeScript|Go|Rust|Ruby|PHP|Swift|Kotlin|R|MATLAB|SQL|NoSQL|MongoDB|PostgreSQL|MySQL|HTML|CSS|React|Angular|Vue|Django|Flask|Spring|Node\.js|Express|TensorFlow|PyTorch|Keras|Scikit-learn|Pandas|NumPy|Docker|Kubernetes|Git|AWS|Azure|GCP|Linux|Unix|Bash|Shell)\b"
EXPERIENCE_PATTERN=r"(\d+)\s+(years|year)"
DEGREE_PATTERN = r"""
\b(
    # Common abbreviated forms
    (?:B.S.\s\.?\s?[A-Z][a-zA-Z]*|M\.?\s?[A-Z][a-zA-Z]*|Ph\.?D\.?|MBA|LL\.?B\.?|LL\.?M\.?) |
    
    # Bachelor's, Master's, Doctorate, Associate's
    (?:Bachelor(?:’s|\'s)?\s(?:degree\s)?(?:in\s[\w\s,&/()-]+(?:\s(?:or|and)\s[\w\s,&/()-]+)*)?) |
    (?:Master(?:’s|\'s)?\s(?:degree\s)?(?:in\s[\w\s,&/()-]+(?:\s(?:or|and)\s[\w\s,&/()-]+)*)?) |
    (?:Doctor(?:ate| of Philosophy| of [\w\s,&/()-]+)) |
    (?:Associate(?:’s|\'s)?\s(?:degree\s)?(?:in\s[\w\s,&/()-]+(?:\s(?:or|and)\s[\w\s,&/()-]+)*)?) |
    
    # Diploma or Certification programs
    (?:Diploma\s(?:in\s[\w\s,&/()-]+)?) |
    (?:Certificate\s(?:in\s[\w\s,&/()-]+)?) |
    
    # Generic Degree + field
    (?:Degree\s(?:in\s[\w\s,&/()-]+)?)
)\b
"""