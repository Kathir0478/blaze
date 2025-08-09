from flask import Flask, request, jsonify
import fitz 
import os
import uuid
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance,Filter,FieldCondition,MatchValue,PayloadSchemaType
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_groq import ChatGroq
import stringToJSON
import promptTemplate

app = Flask(__name__)

load_dotenv()

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

rondomized_llm = ChatGroq(temperature=0.3,model_name="LLaMA3-8b-8192",groq_api_key=GROQ_API_KEY) 
conditioned_llm = ChatGroq(temperature=0,model_name="LLaMA3-8b-8192",groq_api_key=GROQ_API_KEY) 


splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)


@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "Welcome to AICruit"}),200

@app.route("/analyze-pdf", methods=["POST"])
def analyze_pdf():

    uploaded_file = request.files.get("file")
    requirements = request.form.getlist("requirements")
    record_id = request.form.get("record_id")
    question_count=request.form.get("question_count")

    if not uploaded_file:
        return jsonify({"error": "No file provided"}) , 400

    if not requirements:
        return jsonify({"error": "No requirements provided"}) , 400
    
    if not record_id:
        return jsonify({"error": "No record_id provided"}) , 400
    
    if not question_count:
        return jsonify({"error": "No question_count provided"}) , 400

    file_bytes = uploaded_file.read()
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = "".join([page.get_text() for page in doc])
    joined_requirements = "\n".join(f"- {r}" for r in requirements)
    record_id=int(record_id)

    formatted_prompt = promptTemplate.analyze_pdf_prompt.format(
        resume=text, 
        requirements=joined_requirements,
        question_count=question_count
    )
    
    response = conditioned_llm.invoke(formatted_prompt)

    parsed = stringToJSON.analyze_pdf_json_converter(response.content.strip())

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
        "score": parsed.get("score")
    }) , 200

@app.route("/technical-round",methods=["POST"])
def technical_round():
    data = request.json
    list_of_domains = data.get("list_of_domains")
    question_count = data.get("question_count")
    allowed_types = data.get("allowed_types")
    difficulty_level = data.get("difficulty_level")

    if not list_of_domains or question_count <= 0 or not allowed_types or not difficulty_level:
        return jsonify({"error": "Invalid input"}) , 400

    list_of_domains=", ".join(list_of_domains)

    formatted_prompt = promptTemplate.technical_round_prompt.format(
        list_of_domains=list_of_domains,
        question_count=question_count,
        allowed_types=allowed_types,
        difficulty_level=difficulty_level
    )

    response = rondomized_llm.invoke(formatted_prompt)

    parsed = stringToJSON.technical_round_json_converter(response.content.strip())

    return jsonify({
        "questions": parsed
    }) , 200

@app.route("/aptitude-round",methods=["POST"])
def aptitude_round():

    data = request.json
    question_count = data.get("question_count")
    difficulty_level = data.get("difficulty_level")

    if question_count <= 0 or not difficulty_level:
        return jsonify({"error": "Invalid input"}) , 400
    
    formatted_prompt = promptTemplate.aptitude_round_prompt.format(
        question_count=question_count,
        difficulty_level=difficulty_level
    )

    response = rondomized_llm.invoke(formatted_prompt)
    
    parsed = stringToJSON.aptitude_round_json_converter(response.content.strip())

    return jsonify({
        "questions": parsed
    }) , 200

@app.route("/problem-round", methods=["POST"])
def problem_round():

    data = request.json
    question_count = data.get("question_count")
    difficulty_rating = data.get("difficulty_rating")

    if question_count <= 0 or not difficulty_rating:
        return jsonify({"error": "Invalid input"}) , 400

    formatted_prompt = promptTemplate.problem_round_prompt.format(
        question_count=question_count, 
        difficulty_rating=difficulty_rating
    )

    response = rondomized_llm.invoke(formatted_prompt)

    parsed = stringToJSON.problem_round_json_converter(response.content.strip())

    return jsonify({
        "questions": parsed
    }) , 200

@app.route("/validate-problem", methods=["POST"])
def validate_problem():

    data = request.json
    code_snippet = data.get("code_snippet")
    problem_statement = data.get("problem_statement")
    language = data.get("language")

    if not code_snippet or not problem_statement or not language:
        return jsonify({"error": "Invalid input"}) , 400

    formatted_prompt = promptTemplate.validate_problem_prompt.format(
        code_snippet=code_snippet,
        problem_statement=problem_statement,
        language=language
    )

    response = conditioned_llm.invoke(formatted_prompt)

    parsed= stringToJSON.validate_problem_json_converter(response.content.strip())

    return jsonify({
        "report": parsed.get("report"),
        "score": parsed.get("score"),
        "errors": parsed.get("errors")
    }) , 200

@app.route("/communication-round",methods=["POST"])
def communication_round():
    data = request.json
    record_id = data.get("record_id")

    if not record_id:
        return jsonify({"error": "Provide record_id"}) , 400
    
    result = qdrant.retrieve(
        collection_name=TASK_COLLECTION,
        ids=[record_id]
    )

    if result is None:
        return jsonify({"error": "Point not found","status":404})
    
    return jsonify({
        "communication_qn":result[0].payload.get("communications")
    }) , 200

@app.route("/communication-report",methods=["POST"])
def communication_report():

    data = request.json
    communications=data.get("communications",[])
    text=data.get("text")

    if not communications or not text:
        return jsonify({"error": "point_id is required"}) , 400

    formatted_prompt = promptTemplate.communication_round_report_prompt.format(
        communications=communications,
        text=text
    )

    response = conditioned_llm.invoke(formatted_prompt)
    
    parsed = stringToJSON.communication_report_json_converter(response.content.strip())

    return jsonify({
        "report": parsed.get("report")
    }) , 200

@app.route("/assign-task",methods=["POST"])
def assign_task():
    data = request.json
    record_id = data.get("record_id")

    if not record_id:
        return jsonify({"error": "Prodvide record_id"}) , 400

    result = qdrant.retrieve(
        collection_name=TASK_COLLECTION,
        ids=[record_id]
    )

    if result is None:
        return jsonify({"error": "Point not found"}) , 404
    
    return jsonify({
        "task": result[0].payload.get("task"),
        "description": result[0].payload.get("description"),
        "requirements": result[0].payload.get("requirements"),
        "deliverables": result[0].payload.get("deliverables")
    }) , 200

@app.route("/create-chatbot", methods=["POST"])
def create_chatbot():
    data = request.json
    websites = data.get("websites", [])
    record_id = data.get("record_id")

    if not websites or not record_id:
        return jsonify({"error": "Missing websites or user_id"}) , 400

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

    return jsonify({"points_added": len(points)}) , 200

@app.route("/query-chatbot", methods=["POST"])
def query_chatbot():
    data = request.json
    record_id = data.get("record_id")
    query = data.get("query")

    if not record_id or not query:
        return jsonify({"error": "Invalid input"}) , 400

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

    formatted_prompt = promptTemplate.query_chatbot_prompt.format(
        user_query=query,
        context=context
    )

    response = rondomized_llm.invoke(formatted_prompt)

    parsed= stringToJSON.llm_response_json_converter(response.content.strip())

    return jsonify({
        "response": parsed.get("response")
    }) , 200

@app.route("/report-generator", methods=["POST"])
def report_generator():

    data = request.json
    record_id = data.get("record_id")
    coded_snippet= data.get("coded_snippet")

    if not record_id or not coded_snippet:
        return jsonify({"error": "Invalid input"}) , 400

    task = qdrant.retrieve(
        collection_name=TASK_COLLECTION,
        ids=[record_id]
    )
    if not task:
        return jsonify({"error": "Point not found"}) , 404 

    task_data = task[0].payload

    formatted_prompt = promptTemplate.report_prompt.format(
        task=task_data.get("task"),
        description=task_data.get("description"),
        requirements=task_data.get("requirements"),
        deliverables= task_data.get("deliverables"),
        code_snippet=coded_snippet
    )

    response = conditioned_llm.invoke(formatted_prompt)
    
    parsed = stringToJSON.report_response_json_converter(response.content.strip())

    return jsonify({
        "report": parsed.get("report"),
        "score": parsed.get("score")
    }) , 200

if __name__ == "__main__":
    app.run(debug=True)
