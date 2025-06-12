from flask import Flask, request, jsonify, render_template, session
import os
import uuid
import PyPDF2
import openai
import pinecone
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from IPython.display import Markdown
from collections import defaultdict
# from google import genai

app = Flask(__name__)
app.secret_key = os.urandom(24)

load_dotenv()

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# In-memory chat history storage that persists across page refreshes
# Structure: {session_id: [{"role": "user/assistant", "content": "message"}]}
chat_history = defaultdict(list)

# Set your API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize Pinecone
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)

# Connect to Pinecone index
INDEX_NAME = "gemini-rag-embeds"
# Check if the index already exists
if not pc.has_index(INDEX_NAME):
    # Create a new index using create_index_for_model with a supported model
    pc.create_index_for_model(
        name=INDEX_NAME,
        cloud="aws",
        region="us-east-1",
        embed={
            "model": "llama-text-embed-v2",
            "field_map": {"text": "text"}
        }
    )
    print(f"Created new Pinecone index: {INDEX_NAME}")
else:
    print(f"Using existing Pinecone index: {INDEX_NAME}")
INDEX = pc.Index(INDEX_NAME)

# Global variable to track if documents have been uploaded
DOC_UPLOADED = False

# Initialize OpenAI client (May use Gemini compatibility, see: https://ai.google.dev/gemini-api/docs/openai)
CLIENT = openai.OpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file_path):
    text = ""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                text += pdf_reader.pages[page_num].extract_text()
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None


def rag_query(query, top_k=10, session_id='default_session'):
    """
    Perform a RAG query using Pinecone's managed embeddings and OpenAI.
    """
    try:
        # Search Pinecone for relevant contexts
        results = INDEX.search(
            namespace="ns1",
            query={
                "top_k": top_k,
                "inputs": {
                    'text': query
                }
            }
        )
        
        # Extract relevant contexts from the results
        contexts = []
        for hit in results["result"]["hits"]:
            if "text" in hit["fields"]:
                contexts.append(hit["fields"]["text"])
        
        # If no contexts were found, return None to fall back to regular response
        if not contexts:
            # return "No context provided. Please Upload documents"
            contexts.append("There is no context/information found.")
        
        # Combine contexts
        combined_context = "\n\n".join(contexts)
        
        # Get chat history for this session (last 5 exchanges)
        session_history = chat_history.get(session_id, [])[-10:]  # Last 5 exchanges (10 messages)
        history_text = ""
        
        if session_history:
            history_text = "Previous conversation:\n"
            for msg in session_history:
                role = "User" if msg["role"] == "user" else "Assistant"
                history_text += f"{role}: {msg['content']}\n"
            history_text += "\n"
        
        # Create prompt for OpenAI
        prompt = f"""
        Based on the following information and previous conversation, please answer the question.
        
        {history_text}
        Information:
        {combined_context}
        
        Question: {query}
        
        Answer:
        """
        
        # Generate response using OpenAI
        system_message = ""
        if DOC_UPLOADED:
            system_message = """
            You are an Assistant specialized in answering questions based on the uploaded document. Answer in a concise manner, proper, natural, and understandable way
            
            Always greet the user when they greet you.
            
            When you don't understand a query or can't find relevant information, politely reply back.
            
            If user asks detailed explaination, provide Structure detailed explanations with headings, bullet points, or numbered lists where appropriate.

            If the query is Off Topic, politely reply back in negative. Ask user to query only related to what is in documents. Don't use pre-existing knowledge for off-topic tasks. 
            
            You can use your pre-existing knowledge to support user query without asking user, if the user query is somewhat related to what in documents, But clearly mention that you used pre-existing knowledge in that respective and the respective knowledge is not found in document.
            
            IMPORTANT: Never mention you're working from "documents" or "provided information." Answer naturally.
            
            When the user ends the chat or says thanks, respond with a polite farewell.
            """
        else:
            system_message = """
            You are a general assistant that can chat about basic topics.
            
            Always greet the user when they greet you.
            
            If asked about specific documents or information, politely say:
            "I can help with general questions. To answer questions about specific documents, please upload them first."
            
            If asked what you can do, explain you can chat and answer questions once documents are uploaded.
            
            DO NOT mention "documents I've been provided" since no documents exist yet.
            
            When the user ends the chat or says thanks, respond with a polite farewell.

            If user says to you that he has uploaded document, say him that he has not uploaded document and ask politely him to upload document.
            """
            
        response = CLIENT.chat.completions.create(
            model="gemini-2.0-flash",
            messages=[
                {"role": "system", "content": system_message + "\n\nIMPORTANT: Do not use markdown formatting in your responses. Provide plain text only."},
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            
            temperature=0.2,
            top_p=0.95,
            max_tokens=1000, # Type max_tokens when using gemini
            response_format={"type": "text"}
        )
        
        print("Doc uploaded:", DOC_UPLOADED)
        # Just return the raw content - the frontend will handle rendering
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in RAG query: {e}")
        return None
    
# Route to clear chat history explicitly (can be called from frontend)
@app.route('/clear_history', methods=['POST'])
def clear_history():
    session_id = request.cookies.get('session', 'default_session')
    if session_id in chat_history:
        chat_history[session_id] = []
    return jsonify({'success': 'Chat history cleared'})

@app.route('/')
def index():
    global DOC_UPLOADED
    DOC_UPLOADED = False
    try:
        # Check if namespace exists before deleting
        INDEX.describe_index_stats()
        # Only attempt to delete if the index exists
        INDEX.delete(namespace="ns1", delete_all=True)
        print("Deleted previous information from Pinecone namespace 'ns1'.")
    except Exception as e:
        print(f"Note: No existing data to delete or namespace doesn't exist yet: {e}")
    # Don't clear chat history when index page is refreshed
    return render_template('index.html')

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Generate a unique filename
        filename = str(uuid.uuid4()) + '_' + secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Extract text from PDF
        extracted_text = extract_text_from_pdf(file_path)
        if not extracted_text:
            return jsonify({'error': 'Failed to extract text from PDF'}), 400
        
        # Don't store the extracted text in session as it's too large
        # session['pdf_text'] = extracted_text

        # Upload to Pinecone
        try:
            # Assuming 'extracted_text' is the text you want to upload
            text_chunks = [extracted_text[i:i+1000] for i in range(0, len(extracted_text), 1000)]

            # Clear the previous information from Pinecone
            try:
                # Check if namespace exists before deleting
                INDEX.describe_index_stats()
                # Only attempt to delete if the index exists
                INDEX.delete(namespace="ns1", delete_all=True)
                print("Deleted previous information from Pinecone namespace 'ns1'.")
            except Exception as e:
                print(f"Note: No existing data to delete or namespace doesn't exist yet: {e}")
            
            print("Uploading new information in batches...")
            # Process in batches of 90 (below Pinecone's limit of 96)
            batch_size = 90
            for i in range(0, len(text_chunks), batch_size):
                batch = text_chunks[i:i+batch_size]
                
                # Prepare batch data for upsert
                batch_data = [
                    {
                        "_id": f"chunk_{i+j}",
                        'text': chunk
                    } for j, chunk in enumerate(batch)
                ]
                
                # Upsert batch to Pinecone
                INDEX.upsert_records("ns1", batch_data)
                print(f"Uploaded batch {i//batch_size + 1} of {(len(text_chunks) + batch_size - 1)//batch_size}")
        except Exception as e:
            print(f"Error uploading to Pinecone: {e}")
            return jsonify({'error': 'Failed to upload to Pinecone'}), 500
        
        global DOC_UPLOADED
        DOC_UPLOADED = True
        return jsonify({'success': 'PDF uploaded and processed successfully'})
    
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    session_id = request.cookies.get('session', 'default_session')
    
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400
    
    # Add user message to chat history
    chat_history[session_id].append({"role": "user", "content": user_message})
    
    # First try to get a response using RAG with Pinecone
    rag_response = rag_query(user_message, session_id=session_id)
    
    # Add assistant response to chat history
    chat_history[session_id].append({"role": "assistant", "content": rag_response})
    
    # Return both the response and the full chat history
    return jsonify({
        'response': rag_response,
        'history': chat_history[session_id]
    })

if __name__ == '__main__':
    app.run(debug=True)