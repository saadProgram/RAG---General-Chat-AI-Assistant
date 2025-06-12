# PDF-based RAG Chat Application

This is a Flask application that allows users to upload PDF documents, extract text from them, and chat with an AI assistant about the content using Retrieval-Augmented Generation (RAG).

## Features

- PDF document upload and text extraction
- Vector storage with Pinecone
- Chat interface with Google Gemini integration
- Persistent chat history
- Batch processing for large documents
- Real-time responses

## Setup

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Set your API keys in a `.env` file:
   ```
   PINECONE_API_KEY=your-pinecone-api-key
   GEMINI_API_KEY=your-gemini-api-key
   ```

3. Run the application:
   ```
   python app.py
   ```

4. Open your browser and navigate to `http://127.0.0.1:5000/`

## How it Works

1. Upload a PDF document
2. The application extracts text from the PDF and chunks it into smaller segments
3. These chunks are stored in Pinecone vector database in batches (to avoid size limits)
4. When you ask a question, the system:
   - Searches Pinecone for relevant text chunks
   - Combines these chunks as context
   - Includes recent chat history for continuity
   - Generates a response using Google Gemini AI
5. Chat history is maintained in memory during your session

## Technical Details

- Uses Google Gemini AI through OpenAI-compatible API
- Implements batch processing to handle Pinecone's 96-item batch limit
- Maintains chat history in memory with session tracking
- Adapts AI responses based on whether documents are uploaded or not