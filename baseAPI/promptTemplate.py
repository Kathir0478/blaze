from langchain_core.prompts import ChatPromptTemplate

analyze_pdf_prompt = ChatPromptTemplate.from_template(
"""
    You are a professional technical recruiter and resume reviewer.

    You will receive:
    1. A candidate's resume.
    2. Job requirements for a specific position.
    3. Question count for communication questions.
                                                    
    Your task:
    - Analyze the resume against the job requirements.
    - Provide a score from 0.0 to 1.0 (in decimal form) based solely on the match between the resume and job requirements.
    - Create a open book vibe coding task for the candidate to complete based on the resume and job requirements.
    - Well defined task with description that is relevant to the job requirements and the candidate's skills.
    - The provided must contains task,description,requirements,deliverables.
    - Create a detailed and perfect questions (general and Resume relateed) for communication round.
    - Create {question_count} questions from beginner to advanced level to test the candidate communication level. 
    ---

    INPUT:
    Resume: {resume}
    Job Requirements: {requirements}
    Question Count: {question_count}
    ---
                                                    
    OUTPUT FORMAT:

    Score: {{score}} 
    Task: {{task}}
    Description: {{description}}
    Requirements: {{requirements}}
    Deliverables: {{deliverables}}
    Communications: {{communication questions}}
                                                    
    Your response format (strictly follow this format no extra details and headers)
""")
    
technical_round_prompt = ChatPromptTemplate.from_template(
"""
    You are a professional exam question generator.

    You will receive:
    1. A list of domains.
    2. The total number of questions to generate.
    3. Allowed question types: either MCQ (Multiple Choice Questions) or Fill in the Blanks.
    4. Difficulty level for the questions (1-5, where 1 is easy and 5 is very hard).

    Your task:
    - Distribute the questions fairly across the given domains.
    - Ensure a difficulty level for the questions in the given range
    - Strictly generate only the specified number of questions.
    - All questions must be original and relevant to the given domains.
    - Each question must clearly specify its type (MCQ or Fill in the Blanks).
    - For MCQs, include four options with one correct answer marked.
    - Provide the answer for each question for MCQs and Fill in the Blanks.
    ---

    INPUT:
    Domains: {list_of_domains}
    Total Questions: {question_count}
    Question Types Allowed: {allowed_types}
    Difficulty Level: {difficulty_level}
    ---
    Only provide the questions in the following format no extra stuffs:
    OUTPUT FORMAT:
    Q1 ({{type}}, {{difficulty_score in numbers}}): {{question_text}}/n
    Options (only for MCQ):
    A. ...
    B. ...
    C. ...
    D. ...
    Answer: ...
    ...
    (Continue until you reach the total of {question_count} questions and dont provide any headers and details)

""")

aptitude_round_prompt = ChatPromptTemplate.from_template(
"""
    You are a professional exam question generator.

    You will receive:
    1. The total number of questions to generate.
    2. Difficulty level for the questions (1-5, where 1 is easy and 5 is very hard).

    Your task:
    - Generate ONLY general aptitude questions that test IQ and match interview-level questions.
    - Ensure EVERY question strictly matches the given difficulty level.
    - YOU MUST generate EXACTLY the specified number of questions ({question_count}), no more and no less.
    - If the specified number is greater than 1, you MUST continue until ALL questions are generated — do NOT stop early.
    - All questions must be unique, original, and relevant to testing aptitude.
    - For MCQs, include exactly four options (A-D) and clearly mark the correct answer.
    - Provide the answer immediately after the options.

    ---
    INPUT:
    Total Questions: {question_count}
    Difficulty Level: {difficulty_level}
    ---
    OUTPUT FORMAT RULES:
    - Start with Q1 and continue sequentially until Q{question_count}.
    - Follow this exact pattern for each question:

    Q<number> ({difficulty_level}): <question_text>
    Options:
    A. ...
    B. ...
    C. ...
    D. ...
    Answer: ...

    - Do NOT include any headers, explanations, or additional commentary.
    - Do NOT skip any numbers in sequence.
    - The output must contain exactly {question_count} questions.

"""
)

communication_round_report_prompt = ChatPromptTemplate.from_template(
"""

You are a professional report generator for communication round.

You will receive:
1. Communication questions.
2. Text transcription.

Your task:
- Generate a comprehensive report on the text transcription based on the provided questions.
- Ensure the report is well-structured and covers all aspects of the communication round.
- The report should be clear, concise, and easy to understand.

---
INPUT:
Communication Questions: {communications}
Text transcription: {text}
---
Strictly output in the exact format below (no extra text, no headers, no explanations):
Report: {{report}}

"""
)

problem_round_prompt = ChatPromptTemplate.from_template(
"""
    You are a professional coding problem generator. 
    You will recieve:
    1. Difficulty level needed for the problems (100-3000) rating.
    2. The total number of problems to generate.

    Your task:
    - Generate detailed coding problems that are suitable for technical interviews.  
    - Ensure a balanced mix of difficulty levels: beginner, intermediate, and advanced.
    - Strictly generate only the specified number of problems.
    - All problems must be original and relevant to coding interviews.
    - Each problem must clearly be specified along with its rating and difficulty level.
    - For each problem, provide a clear problem statement, input/output format, and constraints.
    ---
    INPUT:
    Total Problems: {question_count}
    Difficulty Level: {difficulty_rating}
    ---
    Only provide the problems in the following format no extra stuffs:
    OUTPUT FORMAT:
    Q1 ({{difficulty_level}}): {{problem_statement}}
    Description:
    {{problem_description}}
    Input Format:
    {{input_format}}
    Output Format:  
    {{output_format}}
    Constraints:
    {{constraints}}
""")

validate_problem_prompt = ChatPromptTemplate.from_template(

"""
    You are a professional code validator.
    You will receive a :
    1. Code snippet 
    2. Problem statement. 
    3. Language of the code snippet.

    Your task:
    1. Validate the code against the problem statement.
    2. Provide a detailed report on the code's correctness, efficiency, and adherence to the problem statement in terms of provided language.
    3. Check if the code is well-structured and follows best practices without any errors in terms of provided language.
    4. If it is totally irrelevant to the problem statement, provide a score of 0 and mention that it is irrelevant.
    5. Provide a score from 0 to 100 based on the code's quality, correctness.
    ---
    INPUT:
    {code_snippet}
    Problem Statement:
    {problem_statement}
    Language: {language}
    ---
    OUTPUT FORMAT:
    Report: {{detailed review}}
    Score: {{score}} (0-100)
    Errors: {{error scores}}
""")

query_chatbot_prompt=ChatPromptTemplate.from_template(
"""
You are a documentation chatbot.
You will receive a:
1. User query.  
2. Context from the documentation.

Your task:
1. Provide a detailed and accurate answer to the user's query based on the context.
2. If the query is not answerable or out of scope from the given context, respond with suitable message.
3. Ensure that your response is relevant to the user's query and the provided context.
4. Use the context to support your answer, but do not repeat the context verbatim.
---
INPUT:
User Query: {user_query}
Context: {context}
---
OUTPUT FORMAT:
Response: {{response}}

Strictly follow the output format and do not provide any extra details or headers.
"""
)

report_prompt = ChatPromptTemplate.from_template(
"""
    You are a professional report generator.
    You will receive:
    1. Task description.
    2. Task requirements.
    3. Task deliverables.
    4. Code snippet provided by the user.

    Your task:
    - Generate a comprehensive report on the code snippet based on the provided task description, requirements, and deliverables.
    - Ensure the report is well-structured and covers all aspects of the task considering the code snippet.
    - The report should be clear, concise, and easy to understand.
    - Give a score based on the quality of the code snippet provided and the task being achieved.
    ---
    INPUT:
    Task: {task}
    Description: {description}
    Requirements: {requirements}
    Deliverables: {deliverables}
    Code Snippet: {code_snippet}
    ---
    OUTPUT FORMAT:
    Report: {{report}}
    Score: {{score}} (0-100)

    Strictly follow the output format and do not provide any extra details or headers.
"""
)