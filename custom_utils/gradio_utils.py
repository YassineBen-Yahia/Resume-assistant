from custom_utils.tools import read_file_tool_func
from agent import agent
def process_inputs(pdf, jd_text, history):

    if pdf is None:
        history += [["System", "❗ Please upload a PDF resume"]]
        return history, "", jd_text
    parts = pdf.name.split("\\")
    pdf_name=parts[-1]

    # Extract resume text using your tool function
    resume_text = read_file_tool_func(pdf_name)["context"]

    history += [["System", "✔ Resume and job description processed. You may now chat with the agent."+pdf.name]]

    return history, resume_text, jd_text, pdf.name




def chat_fn(message, history, resume_text, jd_text, pdf_name):
    print("pdf_name in chat fn:", pdf_name)

    if resume_text == "" or jd_text == "":
        history += [
            ["User", message],
            ["System", "❗ Upload resume + job description first."+pdf_name]
        ]
        return history, resume_text, jd_text
    parts = pdf_name.split("\\")
    pdf=parts[-1]


    # Auto-inject context so user doesn’t repeat it
    system_context = f"""Resume path: '{pdf}' . Job description: '{jd_text}' ."""

    final_input = system_context + message

    result = agent.invoke({"messages": final_input})

    history += [["User", message], ["Agent", result["messages"][-1].content]]


    return history, resume_text, jd_text , pdf_name


