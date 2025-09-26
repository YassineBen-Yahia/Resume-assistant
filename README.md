#  Resume Assistant - AI-Powered CV Analysis & Job Matching

An intelligent web application that analyzes resumes, extracts key information using NLP, and calculates job fit scores to help job seekers optimize their applications.


##  Features

###  **Resume Analysis**
- **PDF Resume Upload**: Drag & drop interface for easy PDF processing
- **Named Entity Recognition (NER)**: Extracts names, skills, experience, education, and contact information
- **Dual Model Approach**: Uses both custom spaCy models and Hugging Face transformers
- **Smart Text Processing**: Handles various resume formats and layouts

###  **Job Matching & Fit Scoring**
- **Intelligent Job Analysis**: Processes job descriptions to extract requirements
- **Comprehensive Fit Score**: Calculates match percentage based on:
  - **Skills (50%)**: Technical and soft skills alignment
  - **Education (25%)**: Degree and qualification matching  
  - **Experience (25%)**: Years of experience comparison


###  **Interactive Chat Interface**
- **Smart Responses**: Rule based chat assistant
- **Real-time Analysis**: Instant job matching when pasting job descriptions
- **Detailed Breakdowns**: Strengths, gaps, and improvement recommendations


###  **Technical Capabilities**
- **Multi-Model NER**: Resume-specific and job-specific entity extraction
- **Semantic Similarity**: TF-IDF vectorization with cosine similarity
- **Data Standardization**: Consistent formatting and deduplication
- **Experience Calculation**: Automatic total experience computation

##  Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YassineBen-Yahia/Resume-assistant.git
   cd Resume-assistant
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python app_flask.py
   ```
   
   Or use the provided batch file:
   ```bash
   run_app.bat
   ```

4. **Access the application**
   Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

##  How to Use

### Step 1: Upload Resume
- Drag and drop your PDF resume onto the upload area
- Or click to browse and select your PDF file
- Wait for the AI analysis to complete

### Step 2: Get Job Match Analysis
- Paste a job description into the chat interface
- The system will automatically detect and analyze the job requirements
- Receive instant fit score and detailed analysis

### Step 3: Review Results
- **Fit Score**: Overall match percentage with letter grade
- **Strengths**: Skills and qualifications that align well
- **Gaps**: Missing requirements and areas for improvement
- **Recommendations**: Specific actions to improve your match

##  Architecture

### Core Components

```
├── app_flask.py          # Main Flask web application
├── fit_calc.py           # Job matching algorithms & scoring
├── utils.py              # Data processing & standardization
├── Process_data.py       # NER entity extraction pipeline
├── templates/
│   └── index.html        # Modern responsive frontend
├── model-best/           # Custom trained spaCy NER model
└── assets/               # Screenshots and documentation
```

### AI Models Used

1. **Custom spaCy Model** [Check the github repository](https://github.com/YassineBen-Yahia/CV-parsing)
   - Trained specifically for resume entity extraction
   - Identifies: Names, Skills, Education, Designations

2. **Hugging Face Models**
   - `manishiitg/resume-ner`: [Resume-specific NER](https://huggingface.co/manishiitg/resume-ner)
   - `Shrav20/job-ner-deberta`: [Job description NER](https://huggingface.co/Shrav20/job-ner-deberta)

3. **Matching Algorithm**
   - TF-IDF vectorization for semantic similarity
   - Weighted scoring system (Skills 50%, Education 25%, Experience 25%)
   - Cosine similarity for skill matching

## 🔬 Technical Details

### Entity Extraction
The system extracts the following entities:

**From Resumes:**
-  **Personal Info**: Names, contact details, email addresses
-  **Skills**: Technical and soft skills with regex patterns
-  **Experience**: Years of experience and job roles
-  **Education**: Degrees, certifications, institutions
-  **Designations**: Current and past job titles

**From Job Descriptions:**
-  **Requirements**: Required skills 
-  **Education**: Degree requirements and preferences
-  **Experience**: Required years of experience

### Scoring Algorithm

```python
def total_match_score(job, candidate):
    skill_score = skill_match_score(job_skills, candidate_skills)      # 50%
    degree_score = degree_match_score(job_degrees, candidate_degrees)  # 25%
    experience_score = experience_match_score(job_exp, candidate_exp)  # 25%
    
    return (0.5 * skill_score) + (0.25 * degree_score) + (0.25 * experience_score)
```

##  Features in Detail

### Fit Score Calculation
- **Skills Matching**: Exact matches + semantic similarity using TF-IDF
- **Education Matching**: Degree level and field alignment
- **Experience Matching**: Years of experience comparison with scaling
- **Comprehensive Analysis**: Detailed breakdown of strengths and gaps

### Data Processing Pipeline
1. **PDF Text Extraction**: PyPDF2 for reliable text parsing
2. **Text Cleaning**: Remove non-ASCII characters, normalize whitespace
3. **Entity Recognition**: Multi-model NER for comprehensive extraction
4. **Data Standardization**: Consistent formatting and deduplication
5. **Score Calculation**: Weighted algorithm for match percentage


##  Future Enhancements

### Planned Features
- [ ] **Multi-language Support**: Support for resumes in different languages
- [ ] **ATS Optimization**: Resume formatting suggestions for ATS systems

### Potential Improvements
- [ ] **Performance Optimization**: Faster processing for large documents
- [ ] **Cloud Deployment**: Scalable cloud infrastructure



##  Screenshots

<div align="center">

| Fit Score Results | Fit Score Results |
|:---:|:---:|
| ![Upload Interface](assets/1.png) | ![Job Analysis](assets/2.png) |

| Skills extraction | Recommendations |
|:---:|:---:|
| ![Fit Score Display](assets/3.png) | ![Analysis Results](assets/4.png) |

</div>
---
