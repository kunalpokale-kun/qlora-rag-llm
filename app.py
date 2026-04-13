import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# -----------------------------
# 🔹 Load Model & Tokenizer
# -----------------------------
@st.cache_resource
def load_model():
    model_path = "qlora-finetuned"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer, device

model, tokenizer, device = load_model()
model.eval()

# -----------------------------
# 🔹 Build Vector DB (RAG)
# -----------------------------
@st.cache_resource
def load_retriever():
    with open("final_dataset.json", "r") as f:
        data = json.load(f)

    documents = []

    for item in data:
        if item.get("instruction") and item.get("output"):
            answer = item["output"].strip()

            # Filter weak data
            if len(answer) > 40:
                documents.append(
                    Document(page_content=answer)
                )

    # 🔹 Chunking (important for RAG quality)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    # 🔹 Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 🔹 Vector DB
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # 🔹 Retriever
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

retriever = load_retriever()

# -----------------------------
# 🔹 Prompt Template (STRICT RAG)
# -----------------------------
def create_prompt(query, context):
    return f"""<s>[INST]
You are a retrieval-based AI assistant.

Answer ONLY using the context below.
Do NOT use outside knowledge.

If answer is not clearly present, say:
"Not found in context."

Context:
{context}

Question:
{query}

Answer:
[/INST]"""

# -----------------------------
# 🔹 RAG Pipeline
# -----------------------------
def rag_pipeline(query):
    docs = retriever.invoke(query)

    if not docs:
        return "No relevant context found.", []

    # ✅ Clean context (ONLY answers)
    context = "\n\n".join([
        doc.page_content.strip()
        for doc in docs
    ])

    # 🔍 DEBUG (remove later if needed)
    print("\n===== CONTEXT USED =====\n", context)

    prompt = create_prompt(query, context)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024   # reduce to avoid overflow
        ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            temperature=0.2,
            top_p=0.9,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 🔹 Clean output
    if "[/INST]" in response:
        response = response.split("[/INST]")[-1].strip()

    return response, docs

# -----------------------------
# 🔹 Streamlit UI
# -----------------------------
st.set_page_config(page_title="Custom LLM + RAG", layout="centered")

st.title("🤖 Custom Fine-Tuned LLM with RAG")
st.write("Ask any technical question")

query = st.text_area("Enter your question:")

if st.button("Generate Answer"):
    if query.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating..."):
            answer, docs = rag_pipeline(query)

        st.subheader("📌 Answer:")
        st.success(answer)

        # 🔹 Retrieved Context
        with st.expander("📚 Retrieved Context"):
            if not docs:
                st.write("No context retrieved.")
            else:
                for i, doc in enumerate(docs):
                    st.markdown(f"**Chunk {i+1}:**")
                    st.write(doc.page_content)