import os 
from dotenv import load_dotenv 
import chromadb 
from openai import OpenAI 
from chromadb.utils import embedding_functions 
import json 
import re 
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai_key, model_name="text-embedding-3-large"
)

chroma_client = chromadb.PersistentClient(path="chroma_persistent_storage")
collection_name = "document_qa_collection"
collection = chroma_client.get_or_create_collection(
    name=collection_name, embedding_function=openai_ef
)

client = OpenAI(api_key=openai_key)

def load_documents_from_directory(directory_path):
    print("==== Loading JSON documents from directory ====")
    try:
        visible = [f for f in os.listdir(directory_path)]
        jsons   = [f for f in visible if f.lower().endswith(".json")]
        print(f"Docs dir: {os.path.abspath(directory_path)}")
        print(f"Found {len(jsons)} JSON files: {jsons}")
    except FileNotFoundError:
        print(f"❌ Documents directory not found: {os.path.abspath(directory_path)}")
        return []
    docs = []
    for filename in os.listdir(directory_path):
        if not filename.lower().endswith(".json"): 
            continue
        path = os.path.join(directory_path, filename)
        try:
            data = json.load(open(path, "r", encoding="utf-8"))
        except Exception as e:
            print(f"⚠️  Skip {filename}: {e}")
            continue

        if isinstance(data, dict) and any(k in data for k in ("majors","minors","tracks")):
            dept = data.get("department") or data.get("metadata", {}).get("department")
            cy   = data.get("catalog_year") or data.get("metadata", {}).get("catalog_year")
            for kind in ("majors","tracks","minors"):
                for rec in data.get(kind, []):
                    prog = _program_name(rec)
                    rid  = f"{filename}::{kind[:-1]}::{_slug(prog)}"
                    docs.append({
                        "id": rid,
                        "text": _flatten_for_rag(rec),
                        "metadata": {"kind": kind[:-1], "department": dept, "program": prog, "catalog_year": cy}
                    })

        # list of items → one doc per item
        elif isinstance(data, list):
            for i, rec in enumerate(data, 1):
                prog = _program_name(rec)
                rid  = f"{filename}::item{i}::{_slug(prog)}"
                docs.append({
                    "id": rid,
                    "text": _flatten_for_rag(rec),
                    "metadata": {"kind": "item", "program": prog, "source_file": filename}
                })

        # single object → one doc
        elif isinstance(data, dict):
            prog = _program_name(data)
            rid  = f"{filename}::root::{_slug(prog)}"
            docs.append({
                "id": rid,
                "text": _flatten_for_rag(data),
                "metadata": {"kind": "object", "program": prog, "source_file": filename}
            })
    print(f"==== Loaded {len(docs)} JSON docs ====")
    return docs
def _program_name(obj):
    if isinstance(obj, dict):
        p = obj.get("program")
        if isinstance(p, dict): 
            return p.get("name") or p.get("short_name") or "unnamed"
        if isinstance(p, str): 
            return p
        for k in ("name","title"): 
            if isinstance(obj.get(k), str): return obj[k]
    return "unnamed"

def _slug(s): 
    return re.sub(r"[^a-z0-9]+","-", str(s).lower()).strip("-")

def _flatten_for_rag(obj):
    # Turn structured JSON into readable, retrieval-friendly text
    def as_text(x):
        if isinstance(x, dict):
            parts=[]
            # preferred high-signal keys first
            prog = x.get("program")
            if isinstance(prog, dict):
                hdr = "Program: " + " | ".join([prog.get("name",""), prog.get("short_name",""), prog.get("degree_code",""), prog.get("department",""), prog.get("faculty","")])
                parts.append(hdr.strip(" |"))
            elif isinstance(prog, str):
                parts.append(f"Program: {prog}")
            for key in ("overview","program_educational_objectives","requirements","ge_requirements","technical_electives","ece_laboratories","ece_restricted_electives","suggested_schedule","tracks","minors","constraints","advising_notes"):
                if key in x: parts.append(f"{key.replace('_',' ').title()}: {x[key]}")
            # generic flatten
            for k,v in x.items():
                if k in {"program","metadata"}: 
                    continue
                if isinstance(v,(dict,list)): 
                    parts.append(as_text(v))
                else: 
                    parts.append(f"{k.replace('_',' ').title()}: {v}")
            return " ".join(str(p) for p in parts if p)
        if isinstance(x, list):
            return " ".join(as_text(i) for i in x)
        return str(x)
    txt = re.sub(r"\s+"," ", as_text(obj)).strip()
    return txt

def split_document(text, chunk_size=1000, chunk_overlap=50):
    anchors = [r"Requirements:", r"Suggested Schedule:", r"General Education", r"Electives", r"Labs", r"Constraints", r"Advising Notes"]
    parts = re.split("|".join(anchors), text, flags=re.I)
    if len(parts) <= 1:  # no anchors found
        return _size_chunks(text, chunk_size, chunk_overlap)
    chunks=[]
    for p in parts:
        chunks.extend(_size_chunks(p, chunk_size, chunk_overlap))
    return chunks

def _size_chunks(text, chunk_size=1000, chunk_overlap=50):
    out=[]; start=0
    while start < len(text):
        end = start + chunk_size
        out.append(text[start:end])
        start = end - chunk_overlap
    return out


# --- Keep your Chroma collection WITH embedding_function (no manual embeddings) ---

# --- FIX your query_documents (Chroma expects a list for query_texts) ---
def query_documents(question, n_results=20):
    results = collection.query(query_texts=[question], n_results=n_results)
    # flatten
    if not results.get("documents"):
        return []

    return [doc for sub in results["documents"] for doc in sub]


# Function to generate a response from OpenAI
def generate_response(question, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    prompt = (
        "You are an assistant for question-answering tasks. Use the following pieces of "
        "retrieved context to answer the question. If you don't know the answer, say that you "
        "don't know."
        "\n\nContext:\n" + context + "\n\nQuestion:\n" + question
    )

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0.3,

        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": question,
            },
        ],
    )

    return response.choices[0].message.content



def build_index_if_missing(documents_dir: str):
    try:
        doc_count = collection.count()
    except Exception:
        doc_count = 0

    if doc_count > 0:
        print(f"==== Collection already has {doc_count} vectors; skipping ingest ====")
        return

    documents = load_documents_from_directory(documents_dir)
    if not documents:
        print("⚠️ No JSON documents loaded. Skipping index build.")
        return

    print(f"Loaded {len(documents)} documents")

    chunked_ids, chunked_texts, chunked_metas = [], [], []
    print("==== Splitting docs into chunks ====")
    for d in documents:
        chunks = split_document(d["text"])
        for i, ch in enumerate(chunks, 1):
            chunked_ids.append(f"{d['id']}::chunk{i:03d}")
            chunked_texts.append(ch)
            chunked_metas.append(d.get("metadata", {}))

    print(f"Split into {len(chunked_texts)} chunks")
    if not chunked_texts:   # <— GUARD
        print("⚠️ No chunks produced; nothing to add.")
        return

    print("==== Inserting chunks into db ====")
    collection.add(ids=chunked_ids, documents=chunked_texts, metadatas=chunked_metas)
    print("==== Ingest complete ====")



if __name__ == "__main__":
    build_index_if_missing("./documents")

    question = "AI Track: pick one valid combination fulfilling “one from List A”, “one from List A or B”, and “one from A/B/C”"
    relevant_chunks = query_documents(question)
    if not relevant_chunks:
        print("No results yet. Did you place valid .json files under ./documents ?")
    else:
        answer = generate_response(question, relevant_chunks)
        print(answer)

