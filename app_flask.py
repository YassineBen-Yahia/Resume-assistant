from flask import Flask, render_template, request, jsonify
from pathlib import Path
import spacy
import re
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from fit_calc import total_match_score
from utils import standardize_data, generate_general_advice, generate_job_analysis, extract_text_from_pdf
from Process_data import process
import traceback

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Add CORS headers to all responses
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Initialize models
try:
    # Load your custom spaCy model
    nlp = spacy.load("model-best")
    print("Custom spaCy model loaded successfully")
except:
    print("Could not load custom model, using alternative approach")
    nlp = None

# Load resume NER model
resume_model_name = "manishiitg/resume-ner"
resume_tokenizer = AutoTokenizer.from_pretrained(resume_model_name)
resume_model = AutoModelForTokenClassification.from_pretrained(resume_model_name)

# Load job NER model
job_model_name = "Shrav20/job-ner-deberta"
job_tokenizer = AutoTokenizer.from_pretrained(job_model_name)
job_model = AutoModelForTokenClassification.from_pretrained(job_model_name)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/test', methods=['GET'])
def test():
    return jsonify({'success': True, 'message': 'Flask server is running!'})


""" 
def process_resume(resume_text):
    
    # Clean text
    cleaned_text = re.sub(r'[^\x00-\x7F]+', '', resume_text)
    resume_text = cleaned_text.replace('\n',' ').replace('\r','')
    resume_text = re.sub('\s+',' ', resume_text).strip()
    
    entities = {}
    
    # Use spaCy model if available
    if nlp:
        doc = nlp(resume_text)
        for ent in doc.ents:
            if ent.label_ != 'Skills':
                entities[ent.label_] = entities.get(ent.label_, []) + [ent.text]
            else:
                skills = ent.text.split(':')
                for skill_group in skills:
                    skills_list = skill_group.split(',')
                    for skill in skills_list:
                        skill = skill.strip()
                        if len(skill) < 30:
                            entities['Skills'] = entities.get('Skills', []) + [skill]
    
    # Use transformer model for additional extraction
    inputs = resume_tokenizer(
        resume_text,
        return_tensors="pt",
        truncation=True,
        is_split_into_words=False
    )
    
    with torch.no_grad():
        outputs = resume_model(**inputs)
    
    pred_ids = torch.argmax(outputs.logits, dim=2)[0]
    tokens = resume_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    labels = [resume_model.config.id2label[id_.item()] for id_ in pred_ids]
    
    # Extract entities from transformer output
    j = -1
    for i, (token, label) in enumerate(zip(tokens, labels)):
        if j > i:
            continue
        if label != "O":
            span = token
            j = i + 1
            while j < len(labels) and labels[j] == label:
                span += " " + resume_tokenizer.convert_tokens_to_string([tokens[j]])
                j += 1
            
            if label != 'DATE':
                entities[label] = entities.get(label, []) + [span]
    
    # Extract additional information using regex
    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    skills_pattern = r"\b(Python|Java|C\+\+|C#|JavaScript|TypeScript|Go|Rust|Ruby|PHP|Swift|Kotlin|R|MATLAB|SQL|NoSQL|MongoDB|PostgreSQL|MySQL|HTML|CSS|React|Angular|Vue|Django|Flask|Spring|Node\.js|Express|TensorFlow|PyTorch|Keras|Scikit-learn|Pandas|NumPy|Docker|Kubernetes|Git|AWS|Azure|GCP|Linux|Unix|Bash|Shell)\b"
    
    skills = re.findall(skills_pattern, resume_text, re.IGNORECASE)
    emails = re.findall(email_pattern, resume_text)
    
    if emails:
        entities['Email'] = entities.get('Email', []) + emails
    if skills:
        entities['Skills'] = entities.get('Skills', []) + skills
    
    # Remove duplicates
    for key in entities:
        entities[key] = list(set(entities[key]))
    
    return standardize_data(entities)
 """
def process_job(job_text):
    """Process job description and extract requirements"""
    # Extract job requirements using transformer model
    inputs = job_tokenizer(
        job_text,
        return_tensors="pt",
        truncation=True,
        is_split_into_words=False
    )
    
    with torch.no_grad():
        outputs = job_model(**inputs)
    
    pred_ids = torch.argmax(outputs.logits, dim=2)[0]
    tokens = job_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    labels = [job_model.config.id2label[id_.item()] for id_ in pred_ids]
    
    job_entities = {}
    
    j = -1
    for i, (token, label) in enumerate(zip(tokens, labels)):
        if j > i:
            continue
        if label != "O":
            span = token
            j = i + 1
            while j < len(labels) and labels[j][2:] == label[2:]:
                span += " " + job_tokenizer.convert_tokens_to_string([tokens[j]])
                j += 1
            
            clean_label = label[2:] if label.startswith(('B-', 'I-')) else label
            if clean_label != 'DATE':
                job_entities[clean_label] = job_entities.get(clean_label, []) + [span]
    
    # Extract additional information using regex
    skills_pattern = r"\b(Python|Java|C\+\+|C#|JavaScript|TypeScript|Go|Rust|Ruby|PHP|Swift|Kotlin|R|MATLAB|SQL|NoSQL|MongoDB|PostgreSQL|MySQL|HTML|CSS|React|Angular|Vue|Django|Flask|Spring|Node\.js|Express|TensorFlow|PyTorch|Keras|Scikit-learn|Pandas|NumPy|Docker|Kubernetes|Git|AWS|Azure|GCP|Linux|Unix|Bash|Shell)\b"
    experience_pattern = r"(\d+)\s+(years|year)"
    degree_pattern = r"\b(Bachelor(?:'s|\'s)?\s(?:degree\s)?(?:in\s[\w\s,&/()-]+)?|Master(?:'s|\'s)?\s(?:degree\s)?(?:in\s[\w\s,&/()-]+)?|Ph\.?D\.?|MBA|B\.S\.|M\.S\.|B\.A\.|M\.A\.)\b"
    
    skills = re.findall(skills_pattern, job_text, re.IGNORECASE)
    experience = re.findall(experience_pattern, job_text, re.IGNORECASE)
    degrees = re.findall(degree_pattern, job_text, re.IGNORECASE)
    
    if skills:
        job_entities['Skills'] = job_entities.get('Skills', []) + skills
    
    if 'SKILL' in job_entities:
        for skill in job_entities['SKILL']:
            skill = skill.replace('▁', '').strip()
            if skill and skill not in job_entities.get('Skills', []):
                job_entities['Skills'] = job_entities.get('Skills', []) + [skill]
    
    if degrees:
        job_entities['Degree'] = job_entities.get('Degree', []) + [d for d in degrees if len(d) < 100]
    
    if 'EDUCATION' in job_entities:
        for edu in job_entities['EDUCATION']:
            edu = edu.replace('▁', '').strip()
            if edu:
                job_entities['Degree'] = job_entities.get('Degree', []) + [edu]
    
    if experience:
        for exp in experience:
            if int(exp[0]) < 50:
                job_entities['ExperianceYears'] = job_entities.get('ExperianceYears', []) + [f"{exp[0]} {exp[1]}"]
    
    # Remove duplicates
    for key in job_entities:
        job_entities[key] = list(set(job_entities[key]))
    
    return standardize_data(job_entities)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({'error': 'Only PDF files are allowed'}), 400
    
    try:
        # Extract text from PDF
        text = extract_text_from_pdf(file)
        
        
        return jsonify({
            'success': True,
            'message': 'Resume analyzed successfully!',
            'text' : text
        })
        
    except Exception as e:
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500

@app.route('/analyze_job_and_resume', methods=['POST'])
def analyze_job_and_resume():
    try:
        data = request.json
        job_text = data.get('job_description', '')
        resume_data = data.get('resume_data', {})
        
        if not job_text:
            return jsonify({'error': 'No job description provided'}), 400
        
        # Process job description
        resume_entities,job_entities = process(nlp,resume_model, resume_tokenizer, job_model, job_tokenizer ,resume_data,job_text)
       
        
        # Calculate fit score if resume data is available
        fit_score = 0
        print (job_entities)
        print (resume_entities)
        if resume_data:
            print("Calculating fit score...")
            fit_score = total_match_score(job_entities, resume_entities)
            print("Fit score calculated:", fit_score)
        
        return jsonify({
            'success': True,
            'job_requirements': job_entities,
            'resume_data': resume_entities,
            'fit_score': round(fit_score * 100, 1),  # Convert to percentage
            'analysis': generate_job_analysis(job_entities, resume_entities, fit_score)
        })
        
    except Exception as e:
        print("Error during job and resume analysis:", e.__traceback__)
        traceback.print_exc()
        return jsonify({'error': f'Error analyzing job: {str(e)}{e.__traceback__}'}), 500


@app.route('/generate_advice', methods=['POST'])
def generate_advice():
    try:
        data = request.json
        job = data.get('job', [])
        resume_data = data.get('resume_data', [])
        
        
        if not job:
            return jsonify({'error': 'No job skills provided'}), 400
       
        advice = generate_general_advice(job, resume_data)
        return jsonify({
            'success': True,
            'advice': advice
        })
        
    except Exception as e:
        print("Error during job and resume analysis:", e.__traceback__)
        traceback.print_exc()
        return jsonify({'error': f'Error analyzing job: {str(e)}{e.__traceback__}'}), 500




if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)