##Overview
This project implements a Retrieval-Augmented Generation (RAG) system using a fine-tuned Large Language Model (LLM) trained with QLoRA.
It combines semantic search (FAISS) with LLM generation to provide context-aware answers.

The system retrieves relevant information from a dataset and uses it to generate accurate responses, reducing hallucination and improving reliability.

##Features
Fine-tuned LLM using QLoRA
Retrieval-Augmented Generation (RAG) pipeline
Semantic search using FAISS
Efficient embeddings via MiniLM
GPU-optimized inference (PyTorch + Transformers)
Interactive UI using Streamlit
Context-aware answer generation

##How It Works
User enters a query in the Streamlit UI
Query is converted into embeddings
FAISS retrieves top-k relevant chunks
Retrieved context is passed to the LLM
LLM generates answer based on context

##Project Structure
QLoRAandRAG/
│── app.py                  
│── final_dataset.json      
│── Project.ipynb           
│── requirements.txt      
│── README.md      

##Installation
1. Clone Repository
git clone https://github.com/your-username/QLoRAandRAG.git
cd QLoRAandRAG
2. Install Dependencies
pip install -r requirements.txt

##Run the App
streamlit run app.py

##Sample Output
Input: "Explain Machine Learning"
Output: Context-aware explanation generated using retrieved knowledge

##Tech Stack
Python
PyTorch
Hugging Face Transformers
QLoRA (Parameter-Efficient Fine-Tuning)
LangChain
FAISS
Sentence Transformers
Streamlit

##Key Highlights
Reduced hallucination using RAG pipeline
Efficient fine-tuning using QLoRA
Fast semantic retrieval using FAISS
End-to-end deployable AI system

##Note
Model weights are not included due to size limitations
You can load your own fine-tuned model in:
model_path = "qlora-finetuned"

##Future Improvements
Deploy on cloud (AWS / GCP / Hugging Face Spaces)
Add chat history (memory)
Improve retrieval ranking
Support multiple datasets

##Author
Kunal Pokale
M.Tech Robotics & AI

If you like this project
Give it a star ⭐ on GitHub!