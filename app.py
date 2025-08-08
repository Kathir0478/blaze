from flask import Flask, request, jsonify
from flask_cors import CORS
import fitz  # PyMuPDF
import os
import uuid
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance,Filter,FieldCondition,MatchValue
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Qdrant 
from langchain_groq import ChatGroq
import stringtojson
import prompttemplate

load_dotenv()
app = Flask(__name__)

# Load env vars
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_HOST = os.getenv("QDRANT_HOST")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TASK_COLLECTION = os.getenv("TASK_COLLECTION")
BOTS_COLLECTION = os.getenv("BOTS_COLLECTION")

# Init Qdrant client
qdrant = QdrantClient(url=QDRANT_HOST, api_key=QDRANT_API_KEY)

if not qdrant.collection_exists(collection_name=TASK_COLLECTION):
    qdrant.create_collection(collection_name=TASK_COLLECTION,vectors_config=VectorParams(size=768, distance=Distance.COSINE))
if not qdrant.collection_exists(collection_name=BOTS_COLLECTION):
    qdrant.create_collection(collection_name=BOTS_COLLECTION,vectors_config=VectorParams(size=768, distance=Distance.COSINE))
    qdrant.create_payload_index(
        collection_name=BOTS_COLLECTION,
        field_name="point_id",
        field_schema="integer"
    )

# Init Embedding Model
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=GOOGLE_API_KEY)

# chat model
llm = ChatGroq(temperature=0,model_name="LLaMA3-8b-8192",groq_api_key=GROQ_API_KEY)  

# text splitter
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
embed_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=64)
# define vector store
vectorstore=Qdrant(collection_name=TASK_COLLECTION,embeddings=embedding_model,client=qdrant)


@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "Welcome to the PDF Analyzer API"}), 200

@app.route("/analyze-pdf", methods=["POST"])
def analyze_pdf():
    # Embed text chunks
    uploaded_file = request.files['file']
    requirements = request.form.getlist("requirements")
    question_count = request.form.get("question_count")  
    if not uploaded_file or not requirements:
        return jsonify({"error": "No file provided"}), 400
    # reading file
    file_bytes = uploaded_file.read()
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = "".join([page.get_text() for page in doc])
    joined_requirements = "\n".join(f"- {r}" for r in requirements)

    chunks = splitter.split_text(text)
    embeddings = embedding_model.embed_documents(chunks)
    
    # Create a unique vector ID
    vector_id = uuid.uuid4().int >> 64  # Qdrant likes int IDs
    # Store average vector for simplicity (or store each chunk if needed)
    avg_vector = [sum(x) / len(x) for x in zip(*embeddings)]
    #saving vector into qdrant
    
    formatted_prompt = prompttemplate.scoretask_prompt.format(
        resume=text, 
        requirements=joined_requirements,
        question_count=question_count
    )
    
    response = llm.invoke(formatted_prompt)
    parsed = stringtojson.parse_score_and_tasks(response.content.strip())

    qdrant.upload_points(collection_name=TASK_COLLECTION,points=[PointStruct(id=vector_id, vector=avg_vector, 
        payload={
            "requirements": parsed.get("requirements", []),
            "task": parsed.get("task", "No task provided"),
            "description": parsed.get("description", "No description provided"),
            "deliverables": parsed.get("deliverables", [])
        }
    )])
    

    return jsonify({
        "score": parsed.get("score"),
        "qdrant_vector_id": vector_id,
        "status":200
    })

@app.route("/generate-technical",methods=["POST"])
def generate_technical_questions():
    data = request.json
    list_of_domains = data.get("list_of_domains")
    question_count = data.get("question_count")
    allowed_types = data.get("allowed_types")
    difficulty_level = data.get("difficulty_level")

    if not list_of_domains or question_count <= 0:
        return jsonify({"error": "Invalid input"}), 400

    list_of_domains=", ".join(list_of_domains)

    formatted_prompt = prompttemplate.technical_prompt.format(
        list_of_domains=list_of_domains,
        question_count=question_count,
        allowed_types=allowed_types,
        difficulty_level=difficulty_level
    )

    response = llm.invoke(formatted_prompt)
    parsed = stringtojson.parse_question_block(response.content.strip())

    return jsonify({
        "questions": parsed, 
        "status": 200
    })

@app.route("/generate-aptitude",methods=["POST"])
def generate_aptitude_questions():

    data = request.json
    question_count = data.get("question_count")
    allowed_types = data.get("allowed_types")
    difficulty_level = data.get("difficulty_level")

    if question_count <= 0:
        return jsonify({"error": "Invalid input"}), 400
    
    formatted_prompt = prompttemplate.aptitude_prompt.format(
        question_count=question_count,
        allowed_types=allowed_types, 
        difficulty_level=difficulty_level
    )

    response = llm.invoke(formatted_prompt)
    parsed = stringtojson.parse_question_block(response.content.strip())

    return jsonify({
        "questions": parsed, 
        "status": 200
    })

@app.route("/generate-problems", methods=["POST"])
def generate_problems():

    data = request.json
    question_count = data.get("question_count")
    difficulty_level = data.get("difficulty_level")
    formatted_prompt = prompttemplate.problem_prompt.format(question_count=question_count, difficulty_level=difficulty_level)

    response = llm.invoke(formatted_prompt)
    parsed = stringtojson.parse_coding_problems(response.content.strip())

    return jsonify({
        "questions": parsed, 
        "status": 200
    })

@app.route("/validate-code", methods=["POST"])
def validate_code():

    data = request.json
    code_snippet = data.get("code_snippet")
    problem_statement = data.get("problem_statement")
    language = data.get("language")

    if not code_snippet or not problem_statement:
        return jsonify({"error": "Invalid input"}), 400

    formatted_prompt = prompttemplate.validate_prompt.format(
        code_snippet=code_snippet,
        problem_statement=problem_statement,
        language=language
    )

    response = llm.invoke(formatted_prompt)
    # response_data = json.loads(response.content.strip())
    parsed= stringtojson.parse_validate_output(response.content.strip())

    return jsonify({
        "report": parsed.get("report"),
        "score": parsed.get("score"),
        "errors": parsed.get("errors"),
        "status": 200
    })

@app.route("/assign-task",methods=["POST"])
def assign_task():
    data = request.json
    point_id = data.get("point_id")

    result = qdrant.retrieve(
        collection_name=TASK_COLLECTION,
        ids=[point_id]
    )
    if result is None:
        return jsonify({"error": "Point not found"}), 404
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
    user_id = data.get("user_id")

    if not websites or not user_id:
        return jsonify({"error": "Missing websites or user_id"}), 400

    loader = WebBaseLoader(websites)
    docs = loader.load()

    chunks = embed_splitter.split_documents(docs)

    # # Step 4: Embed and upload
    points = []
    for idx, chunk in enumerate(chunks):
        vector = embedding_model.embed_query(chunk.page_content)
        points.append(PointStruct(
            id=uuid.uuid4().int >> 64,
            vector=vector,
            payload={"point_id":int(user_id),"content": chunk.page_content}
        ))

    qdrant.upsert(
        collection_name=BOTS_COLLECTION,
        points=points
    )

    return jsonify({"status": 200,"points_added": len(points)})

@app.route("/ask-query", methods=["POST"])
def ask_query():
    data = request.json
    point_id = data.get("point_id")
    query = data.get("query")

    if not point_id or not query:
        return jsonify({"error": "point_id and query are required"}), 400

    query_vector = embedding_model.embed_query(query)

    search_result = qdrant.search(
        collection_name=BOTS_COLLECTION,
        query_vector=query_vector,
        limit=10,
        query_filter=Filter(
            must=[FieldCondition(
                key="point_id",
                match=MatchValue(value=int(point_id))
            )]
        )
    )

    context_texts = [hit.payload["content"] for hit in search_result]
    context = "\n\n".join(context_texts)
    formatted_prompt = prompttemplate.chat_prompt.format(
        user_query=query,
        context=context
    )
    response = llm.invoke(formatted_prompt)
    parsed= stringtojson.llm_response_to_json(response.content.strip())
    return jsonify({
        "response": parsed.get("response"),
        "status": 200
    })

@app.route("/report-generator", methods=["POST"])
def report_generator():
    data = request.json
    point_id = data.get("point_id")
    coded_snippet= data.get("coded_snippet")
    if not point_id:
        return jsonify({"error": "point_id is required"}), 400

    task = qdrant.retrieve(
        collection_name=TASK_COLLECTION,
        ids=[point_id]
    )
    if not task:
        return jsonify({"error": "Point not found"}), 404

    task_data = task[0].payload

    formatted_prompt = prompttemplate.report_prompt.format(
        task=task_data.get("task"),
        description=task_data.get("description"),
        requirements=task_data.get("requirements"),
        deliverables= task_data.get("deliverables"),
        code_snippet=coded_snippet
    )

    response = llm.invoke(formatted_prompt)
    print(response.content.strip())
    parsed = stringtojson.parse_report_generator_output(response.content.strip())

    return jsonify({
        "report": parsed.get("report"),
        "score": parsed.get("score"),
        "status": 200
    })

if __name__ == "__main__":
    app.run(debug=True)
