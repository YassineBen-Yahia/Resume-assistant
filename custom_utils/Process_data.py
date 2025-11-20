
import re
import torch
from custom_utils.regex_patterns import EMAIL_PATTERN, SKILLS_PATTERN, EXPERIENCE_PATTERN, DEGREE_PATTERN
from custom_utils.fit_calc import total_match_score
from custom_utils.utils import standardize_data, skills_mapping


def extract_same_skills(J, labels, tokens, jobtokenize):
    """Extract entities from job description using Hugging Face NER model
       Example: ▁management (B-EDUCATION)
                ▁Java  (B-SKILL)...
    
    """
    
    j=-1
    for i, (token, label) in enumerate(zip(tokens, labels)):
        if(j>i):
            continue
        if label != "O":
            # Start a span with the current token
            span = token
            j = i + 1
            while j < len(labels) and labels[j][2:] == label[2:]:# Check consecutive tokens with the same label
                span +=" "
                span += jobtokenize.convert_tokens_to_string([tokens[j]])# Concatenate subsequent tokens with the same label
                j += 1
            if label !='DATE':
                J[label[2:]] = J.get(label[2:], []) + [span]
    return J




def process (nlp, ner_resume,tokenizer, ner_job,jobtokenize, resume_text, job_text):


    ###############################################################################################
    # RESUME PROCESSING
    ###############################################################################################


    # 1. Preprocess the input text
    cleaned_text = re.sub(r'[^\x00-\x7F]+', '', resume_text)
    resume_text = cleaned_text.replace('\n',' ').replace('\r','')
    resume_text = re.sub('\s+',' ',resume_text).strip()

    #dictionary to hold the extracted entities
    #1. Extract entities from resume using spaCy NER model
    L = {}
    doc = nlp(resume_text)
    for ent in doc.ents:
        if ent.label_ !='Skills':
            L[ent.label_] = L.get(ent.label_, []) + [ent.text]
        else:
            skills = ent.text.split(':')
            for i in range(len(skills)):
                skills2 = skills[i].split(',')
                for i in range(len(skills2)):
                    skills2[i] = skills2[i].strip()
                    if len(skills2[i]) < 30:
                        L['Skills'] = L.get('Skills', []) + [skills2[i]]

    # Use Hugging Face transformers NER model to extract email and skills
    # 1. Tokenize input text  
    inputs = tokenizer(
        resume_text,
        return_tensors="pt",
        truncation=True,
        is_split_into_words=False
    )

    # 2. Forward pass
    with torch.no_grad():
        outputs = ner_resume(**inputs)          # outputs.logits shape: [batch, seq_len, num_labels]

    # 3. Take the most probable label for each token
    pred_ids = torch.argmax(outputs.logits, dim=2)[0]    # shape: [seq_len]

    # 4. Convert back to words + labels
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    labels = [ner_resume.config.id2label[id_.item()] for id_ in pred_ids]



    j=-1
    for i, (token, label) in enumerate(zip(tokens, labels)):
        if(j>i):
            continue
        if label != "O":
            # Start a span with the current token
            span = token
            j = i + 1
            while j < len(labels) and labels[j] == label:# Check consecutive tokens with the same label
                span +=" "
                span += tokenizer.convert_tokens_to_string([tokens[j]])# Concatenate subsequent tokens with the same label
                j += 1


    
    skills = re.findall(SKILLS_PATTERN, resume_text)
    #print(f"Skills found: {skills}")
    emails = re.findall(EMAIL_PATTERN, resume_text)
    if emails:
        L['Email Address'] = L.get('EMAIL', []) + emails
    if skills:
        L['Skills'] = L.get('Skills', []) + skills   

    d_pattern = re.compile(DEGREE_PATTERN, re.IGNORECASE | re.VERBOSE) 

    for key in L:
        L[key] = list(set(L[key]))  # Remove duplicates




    print(f"Resume Skills found: {L.get('Skills', [])}")
    print()
    print("********************************")
    print() 

    ###############################################################################################
    # JOB DESCRIPTION PROCESSING
    ###############################################################################################

    inputs = jobtokenize(
        job_text,
        return_tensors="pt",
        truncation=True,
        is_split_into_words=False
    )

    # 2. Forward pass
    with torch.no_grad():
        outputs = ner_job(**inputs)          # outputs.logits shape: [batch, seq_len, num_labels]

    # 3. Take the most probable label for each token
    pred_ids = torch.argmax(outputs.logits, dim=2)[0]    # shape: [seq_len]

    # 4. Convert back to words + labels
    tokens = jobtokenize.convert_ids_to_tokens(inputs["input_ids"][0])
    labels = [ner_job.config.id2label[id_.item()] for id_ in pred_ids]

    J={}
    J=extract_same_skills(J, labels, tokens, jobtokenize)


    jobskills = re.findall(SKILLS_PATTERN, job_text)
    degrees= re.findall(d_pattern, job_text)
    experience = re.findall(EXPERIENCE_PATTERN, job_text)


    Job={}
    if jobskills:
        Job['Skills'] = Job.get('Skills', []) + jobskills
    if 'SKILL' in J:
        for skill in J['SKILL']:
            if skill not in Job['Skills']:
                if skill[0]=='▁':
                    skill=skill[1:]
                Job['Skills'] = Job.get('Skills', []) + [skill]

    
    """
    if 'Skills' in Job:
        Job['Skills'] = skills_mapping(Job['Skills'])
    if 'Skills' in L:
        L['Skills'] = skills_mapping(L['Skills'])
    """

    for d in degrees:
        if "degree" in d.lower() or "bachelor" in d.lower() or "master" in d.lower() or "doctor" in d.lower() or "associate" in d.lower() or "diploma" in d.lower() or "certificate" in d.lower():
            if len(d)>60:
                Job['Degree'] = Job.get('Degree', []) + [d[:60]]
            else:
                Job['Degree'] = Job.get('Degree', []) + [d]


    if 'EDUCATION' in J:
        for skill in J['EDUCATION']:
            if 'Degree' in Job and skill not in Job['Degree']:
                if skill[0]=='▁':
                    skill=skill[1:]
                Job['Degree'] = Job.get('Degree', []) + [skill]

    if experience:
        for exp in experience:
            if int(exp[0]) < 50:  # Filter out unrealistic experience values
                Job['ExperianceYears'] = Job.get('ExperianceYears', []) + [f"{exp[0]} {exp[1]}"]  
    Job=standardize_data(Job)
    L=standardize_data(L)

        
    print(f"Total Match Score: {total_match_score(Job, L)}")
    for key in Job:
        print(f"{key}: {Job[key]}")

    print()
    print("********************************")
    print()

    for key in L:
        print(f"{key}: {L[key]}")
    
   
    return L, Job




