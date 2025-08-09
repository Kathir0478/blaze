from flask import Flask, request, jsonify
import fitz  # PyMuPDF
import os
import uuid
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance,Filter,FieldCondition,MatchValue,PayloadSchemaType
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Qdrant 
from langchain_groq import ChatGroq
import stringtojson
import prompttemplate

load_dotenv()
app = Flask(__name__)

QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_HOST = os.getenv("QDRANT_HOST")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TASK_COLLECTION = os.getenv("TASK_COLLECTION")
BOTS_COLLECTION = os.getenv("BOTS_COLLECTION")

qdrant = QdrantClient(url=QDRANT_HOST, api_key=QDRANT_API_KEY)

if not qdrant.collection_exists(collection_name=TASK_COLLECTION):
    qdrant.create_collection(collection_name=TASK_COLLECTION,vectors_config=VectorParams(size=768, distance=Distance.COSINE))

if not qdrant.collection_exists(collection_name=BOTS_COLLECTION):
    qdrant.create_collection(collection_name=BOTS_COLLECTION,vectors_config=VectorParams(size=768, distance=Distance.COSINE))
    qdrant.create_payload_index(
        collection_name=BOTS_COLLECTION,
        field_name="record_id",
        field_schema=PayloadSchemaType.INTEGER
    )

embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=GOOGLE_API_KEY)

llm = ChatGroq(temperature=0,model_name="LLaMA3-8b-8192",groq_api_key=GROQ_API_KEY)  

splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)

@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "Welcome to the PDF Analyzer API","status": 200})

@app.route("/analyze-pdf", methods=["POST"])
def analyze_pdf():

    uploaded_file = request.files['file']
    requirements = request.form.getlist("requirements")
    record_id = request.form.get("record_id")
    record_id = int(record_id)
    question_count=request.form.get("question_count")

    if not uploaded_file or not requirements:
        return jsonify({"error": "No file provided","status":400})

    file_bytes = uploaded_file.read()
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = "".join([page.get_text() for page in doc])
    joined_requirements = "\n".join(f"- {r}" for r in requirements)

    formatted_prompt = prompttemplate.analyze_pdf_prompt.format(
        resume=text, 
        requirements=joined_requirements,
        question_count=question_count
    )
    
    response = llm.invoke(formatted_prompt)

    parsed = stringtojson.analyze_pdf_json_converter(response.content.strip())

    qdrant.upload_points(collection_name=TASK_COLLECTION,points=[PointStruct(id=record_id,vector=embedding_model.embed_query(text),
        payload={
            "requirements": parsed.get("requirements", []),
            "task": parsed.get("task", "No task provided"),
            "description": parsed.get("description", "No description provided"),
            "deliverables": parsed.get("deliverables", []),
            "communications": parsed.get("communications",[])
        }
    )])
    

    return jsonify({
        "score": parsed.get("score"),
        "status":200
    })

@app.route("/technical-round",methods=["POST"])
def technical_round():
    data = request.json
    list_of_domains = data.get("list_of_domains")
    question_count = data.get("question_count")
    allowed_types = data.get("allowed_types")
    difficulty_level = data.get("difficulty_level")

    if not list_of_domains or question_count <= 0:
        return jsonify({"error": "Invalid input","status":400})

    list_of_domains=", ".join(list_of_domains)

    formatted_prompt = prompttemplate.technical_round_prompt.format(
        list_of_domains=list_of_domains,
        question_count=question_count,
        allowed_types=allowed_types,
        difficulty_level=difficulty_level
    )

    response = llm.invoke(formatted_prompt)

    parsed = stringtojson.technical_round_json_converter(response.content.strip())

    return jsonify({
        "questions": parsed, 
        "status": 200
    })

@app.route("/aptitude-round",methods=["POST"])
def aptitude_round():

    data = request.json
    question_count = data.get("question_count")
    difficulty_level = data.get("difficulty_level")

    if question_count <= 0:
        return jsonify({"error": "Invalid input","status":400})
    
    formatted_prompt = prompttemplate.aptitude_round_prompt.format(
        question_count=question_count,
        difficulty_level=difficulty_level
    )

    response = llm.invoke(formatted_prompt)
    print(response.content.strip())
    parsed = stringtojson.aptitude_round_json_converter(response.content.strip())

    return jsonify({
        "questions": parsed, 
        "status": 200
    })

@app.route("/problem-round", methods=["POST"])
def problem_round():

    data = request.json
    question_count = data.get("question_count")
    difficulty_rating = data.get("difficulty_rating")

    formatted_prompt = prompttemplate.problem_round_prompt.format(
        question_count=question_count, 
        difficulty_rating=difficulty_rating
    )

    response = llm.invoke(formatted_prompt)

    parsed = stringtojson.problem_round_json_converter(response.content.strip())

    return jsonify({
        "questions": parsed, 
        "status": 200
    })

@app.route("/validate-problem", methods=["POST"])
def validate_problem():

    data = request.json
    code_snippet = data.get("code_snippet")
    problem_statement = data.get("problem_statement")
    language = data.get("language")

    if not code_snippet or not problem_statement:
        return jsonify({"error": "Invalid input","status":400})

    formatted_prompt = prompttemplate.validate_problem_prompt.format(
        code_snippet=code_snippet,
        problem_statement=problem_statement,
        language=language
    )

    response = llm.invoke(formatted_prompt)

    parsed= stringtojson.validate_problem_json_converter(response.content.strip())

    return jsonify({
        "report": parsed.get("report"),
        "score": parsed.get("score"),
        "errors": parsed.get("errors"),
        "status": 200
    })

@app.route("/communication-round",methods=["POST"])
def communication_round():
    data = request.json
    record_id = data.get("record_id")

    result = qdrant.retrieve(
        collection_name=TASK_COLLECTION,
        ids=[record_id]
    )

    if result is None:
        return jsonify({"error": "Point not found","status":404})
    
    return jsonify({
        "communication_qn":result[0].payload.get("communications"),
        "status": 200
    })

@app.route("/communication-report",methods=["POST"])
def communication_report():

    data = request.json
    communications=data.get("communications",[])
    text=data.get("text")

    if not communications or not text:
        return jsonify({"error": "point_id is required","status":400})

    formatted_prompt = prompttemplate.communication_round_report_prompt.format(
        communications=communications,
        text=text
    )

    response = llm.invoke(formatted_prompt)
    
    parsed = stringtojson.communication_report_json_converter(response.content.strip())

    return jsonify({
        "report": parsed.get("report"),
        "status": 200
    })

@app.route("/assign-task",methods=["POST"])
def assign_task():
    data = request.json
    record_id = data.get("record_id")

    result = qdrant.retrieve(
        collection_name=TASK_COLLECTION,
        ids=[record_id]
    )

    if result is None:
        return jsonify({"error": "Point not found","status":404})
    
    return jsonify({
        "task": result[0].payload.get("task"),
        "description": result[0].payload.get("description"),
        "requirements": result[0].payload.get("requirements"),
        "deliverables": result[0].payload.get("deliverables"),
        "status": 200
    })

@app.route("/create-chatbot", methods=["POST"])
def create_chatbot():
    data = request.json
    websites = data.get("websites", [])
    record_id = data.get("record_id")

    if not websites or not record_id:
        return jsonify({"error": "Missing websites or user_id","status":400})

    loader = WebBaseLoader(websites)
    docs = loader.load()

    chunks = splitter.split_documents(docs)

    points = []
    for chunk in chunks:
        vector = embedding_model.embed_query(chunk.page_content)
        points.append(PointStruct(
            id=uuid.uuid4().int >> 64,
            vector=vector,
            payload={"record_id":int(record_id),"content": chunk.page_content}
        ))

    qdrant.upsert(
        collection_name=BOTS_COLLECTION,
        points=points
    )

    return jsonify({"status": 200,"points_added": len(points)})

@app.route("/query-chatbot", methods=["POST"])
def query_chatbot():
    data = request.json
    record_id = data.get("record_id")
    query = data.get("query")

    if not record_id or not query:
        return jsonify({"error": "point_id and query are required","status":400})

    query_vector = embedding_model.embed_query(query)

    search_result = qdrant.search(
        collection_name=BOTS_COLLECTION,
        query_vector=query_vector,
        limit=15,
        query_filter=Filter(
            must=[FieldCondition(
                key="record_id",
                match=MatchValue(value=int(record_id))
            )]
        )
    )

    context_texts = [hit.payload["content"] for hit in search_result]
    context = "\n\n".join(context_texts)

    formatted_prompt = prompttemplate.query_chatbot_prompt.format(
        user_query=query,
        context=context
    )

    response = llm.invoke(formatted_prompt)

    parsed= stringtojson.llm_response_json_converter(response.content.strip())

    return jsonify({
        "response": parsed.get("response"),
        "status": 200
    })

@app.route("/report-generator", methods=["POST"])
def report_generator():

    data = request.json
    record_id = data.get("record_id")
    coded_snippet= data.get("coded_snippet")

    if not record_id:
        return jsonify({"error": "point_id is required","status":400})

    task = qdrant.retrieve(
        collection_name=TASK_COLLECTION,
        ids=[record_id]
    )
    if not task:
        return jsonify({"error": "Point not found","status":404})

    task_data = task[0].payload

    formatted_prompt = prompttemplate.report_prompt.format(
        task=task_data.get("task"),
        description=task_data.get("description"),
        requirements=task_data.get("requirements"),
        deliverables= task_data.get("deliverables"),
        code_snippet=coded_snippet
    )

    response = llm.invoke(formatted_prompt)
    
    parsed = stringtojson.report_response_json_converter(response.content.strip())

    return jsonify({
        "report": parsed.get("report"),
        "score": parsed.get("score"),
        "status": 200
    })

if __name__ == "__main__":
    app.run(debug=True)
