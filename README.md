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

##  Quick Start


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
├── utils.py              # Data procesing and advice generation
├── Process_data.py       # NER entity extraction pipeline
├── templates/
│   └── index.html        # Modern responsive frontend
├── model-best/           # Custom trained spaCy NER model
└── assets/               # Screenshots and documentation
```


##  Technical Details

### Entity Extraction
The system extracts the following entities:

**From Resumes:**
-  **Personal Info - Skills - Experience - Degree - Designation**

**From Job Descriptions:**
-  **Requirements**: Required skills 
-  **Education**: Degree requirements and preferences
-  **Experience**: Required years of experience



##  Future Enhancements

### Potential Improvements
- [ ] **Mapping related skills**: For exapmle map "Deep Learning" --> "PyTorch" or "Tensorflow"
- [ ] **Add a LLM layer**: More accurate chatbot


## Some Problems

The pipeline I designed doesn't always get an accurate score for example when the skills required are "Python" "Pandas" or "XGBoost experience" but the resume just mentions "Machine learning skills" it gives a low score even with semantic similarities.  



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
