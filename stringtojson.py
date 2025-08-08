import re

def parse_validate_output(text: str) -> dict:
    result = {
        "report": None,
        "score": None,
        "errors": None
    }

    # Extract Report
    report_match = re.search(r"Report:\s*(.*?)(?=Score:|Errors:|$)", text, re.DOTALL)
    if report_match:
        result["report"] = report_match.group(1).strip()

    # Extract Score
    score_match = re.search(r"Score:\s*(\d+)", text)
    if score_match:
        result["score"] = int(score_match.group(1))

    # Extract Errors
    errors_match = re.search(r"Errors:\s*(.*)", text, re.DOTALL)
    if errors_match:
        result["errors"] = errors_match.group(1).strip()

    return result

def parse_question_block(raw_output: str):
    questions = []
    
    # Find all Qn blocks using regex pattern
    q_blocks = re.findall(r"(Q\d+\s+\([^)]+\):.*?(?=(?:\nQ\d+\s+\([^)]+\):)|\Z))", raw_output.strip(), re.DOTALL)

    for block in q_blocks:
        lines = block.strip().split('\n')
        if not lines or not lines[0].startswith("Q"):
            continue

        # Parse first line: Q1 (MCQ, 2): question text
        match = re.match(r"Q(\d+)\s+\(([^,]+),\s*([^)]+)\):\s+(.*)", lines[0])
        if not match:
            continue

        q_num = int(match.group(1))
        q_type = match.group(2).strip()
        difficulty = match.group(3).strip()
        question_text = match.group(4).strip()

        options = []
        answer = None

        # Parse rest of the block
        for line in lines[1:]:
            line = line.strip()
            if re.match(r"^[A-D]\.", line):
                options.append(line[3:].strip())  # Remove "A. " etc.
            elif line.startswith("Answer:"):
                answer = line.split(":", 1)[1].strip()
                if q_type == "MCQ" and answer and answer[1:3] == ". ":
                    answer = answer[3:]  # Remove "A. ..." to keep just option text

        q_obj = {
            "question_number": q_num,
            "type": q_type,
            "difficulty": difficulty,
            "question": question_text,
            "options": options if q_type == "MCQ" else None,
            "answer": answer
        }
        questions.append(q_obj)

    return questions

def parse_coding_problems(text: str):
    problems = []
    
    # Regex to split the questions
    question_blocks = re.split(r'\n(?=Q\d+ \()', text.strip())
    
    for block in question_blocks:
        # Extract question ID and difficulty
        header_match = re.match(r'Q(\d+) \((.*?)\): (.*)', block)
        if not header_match:
            continue
        
        q_id = f"Q{header_match.group(1)}"
        difficulty = header_match.group(2).strip()
        statement = header_match.group(3).strip()
        
        # Extract other fields
        description_match = re.search(r'Description:\n(.*?)\nInput Format:', block, re.DOTALL)
        input_format_match = re.search(r'Input Format:\n(.*?)\nOutput Format:', block, re.DOTALL)
        output_format_match = re.search(r'Output Format:\n(.*?)\nConstraints:', block, re.DOTALL)
        constraints_match = re.search(r'Constraints:\n(.*)', block, re.DOTALL)

        problem = {
            "id": q_id,
            "difficulty_level": difficulty,
            "problem_statement": statement,
            "problem_description": description_match.group(1).strip() if description_match else "",
            "input_format": input_format_match.group(1).strip() if input_format_match else "",
            "output_format": output_format_match.group(1).strip() if output_format_match else "",
            "constraints": constraints_match.group(1).strip() if constraints_match else ""
        }

        problems.append(problem)

    return problems

def parse_score_and_tasks(llm_response: str):

    if not llm_response or not isinstance(llm_response, str):
        return {
            "score": None,
            "task": None,
            "description": None,
            "requirements": [],
            "deliverables": []
        }

    # Extract score
    score_match = re.search(r"Score\s*:\s*([0-1](?:\.\d+)?|\.\d+)", llm_response, re.IGNORECASE)
    score_val = float(score_match.group(1)) if score_match else None

    # Extract task
    task_match = re.search(r"Task\s*:\s*(.+?)(?=\n\s*Description\s*:|$)", llm_response, re.IGNORECASE | re.DOTALL)
    task_text = task_match.group(1).strip() if task_match else None

    # Extract description
    desc_match = re.search(r"Description\s*:\s*(.+?)(?=\n\s*Requirements\s*:|$)", llm_response, re.IGNORECASE | re.DOTALL)
    description_text = desc_match.group(1).strip() if desc_match else None

    # Extract requirements as list
    req_match = re.search(r"Requirements\s*:\s*(.+?)(?=\n\s*Deliverables\s*:|$)", llm_response, re.IGNORECASE | re.DOTALL)
    requirements_list = []
    if req_match:
        raw_reqs = req_match.group(1).strip()
        requirements_list = [line.lstrip("*").strip() for line in raw_reqs.splitlines() if line.strip().startswith("*")]

    # Extract deliverables as list
    deliv_match = re.search(r"Deliverables\s*:\s*(.+)", llm_response, re.IGNORECASE | re.DOTALL)
    deliverables_list = []
    if deliv_match:
        raw_delivs = deliv_match.group(1).strip()
        deliverables_list = [line.lstrip("*").strip() for line in raw_delivs.splitlines() if line.strip().startswith("*")]

    return {
        "score": score_val,
        "task": task_text,
        "description": description_text,
        "requirements": requirements_list,
        "deliverables": deliverables_list
    }

def llm_response_to_json(llm_text: str) -> dict:
    match = re.match(r"^Response:\s*(.*)", llm_text.strip(), re.IGNORECASE | re.DOTALL)
    if match:
        response_content = match.group(1).strip()
    else:
        response_content = llm_text.strip() 

    return {"response": response_content}

def parse_report_generator_output(llm_text: str) -> dict:
    text = llm_text.strip()

    pattern = r"^Report:\s*(.*?)\s*Score:\s*(\d{1,3})\s*(?:\(0-100\))?$"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)

    if not match:
        raise ValueError("LLM output is not in the expected format.")

    report_content = match.group(1).strip()
    score_value = int(match.group(2))

    return {
        "report": report_content,
        "score": score_value
    }