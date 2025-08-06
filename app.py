from flask import Flask, request, jsonify
import fitz  # PyMuPDF
import os
import uuid
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Qdrant 
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

load_dotenv()
app = Flask(__name__)

# Load env vars
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_HOST = os.getenv("QDRANT_HOST")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# Init Qdrant client
qdrant = QdrantClient(url=QDRANT_HOST, api_key=QDRANT_API_KEY)
collection_name = "pdf_vectors"

# Ensure collection exists
qdrant.recreate_collection(collection_name=collection_name,vectors_config=VectorParams(size=768, distance=Distance.COSINE))

# Init Embedding Model
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=GOOGLE_API_KEY)
# chat model
llm = ChatGroq(temperature=0.2,model_name="LLaMA3-8b-8192",groq_api_key=GROQ_API_KEY)  
# text splitter
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# define vector store
vectorstore=Qdrant(collection_name=collection_name,embeddings=embedding_model,client=qdrant)

@app.route("/analyze-pdf", methods=["POST"])
def analyze_pdf():
    #file handling
    uploaded_file = request.files['file']
    requirements = request.form.getlist("requirements")
    if not uploaded_file or not requirements:
        return jsonify({"error": "Missing file or requirements"}), 400
    # reading file
    file_bytes = uploaded_file.read()
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = "".join([page.get_text() for page in doc])

    # Embed text chunks
    chunks = splitter.split_text(text)
    embeddings = embedding_model.embed_documents(chunks)

    # Create a unique vector ID
    vector_id = uuid.uuid4().int >> 64  # Qdrant likes int IDs
    # Store average vector for simplicity (or store each chunk if needed)
    avg_vector = [sum(x) / len(x) for x in zip(*embeddings)]
    #saving vector into qdrant
    qdrant.upload_points(collection_name=collection_name,points=[PointStruct(id=vector_id, vector=avg_vector)])
    
    joined_requirements = "\n".join(f"- {r}" for r in requirements)
    score_prompt = ChatPromptTemplate.from_template("""
        You are a professional resume reviewer.

        Here is a resume:
        ---
        {resume}
        ---

        Here are the job requirements:
        ---
        {requirements}
        ---

        Give a score from 0 to 1 in decimal based on how well the resume matches the requirements. Be objective and provide only the score value.
    """)  
    
    formatted_prompt = score_prompt.format(resume=text, requirements=joined_requirements)
    response = llm.invoke(formatted_prompt)
    
    return jsonify({
        "score": float(response.content.strip()),
        "qdrant_vector_id": vector_id,
        "status":200
    })

if __name__ == "__main__":
    app.run(debug=True)
