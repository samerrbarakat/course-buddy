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

def generate_multi_query(query, model="gpt-3.5-turbo"):
    """
    Take a single query and ask the LLM to generate several related questions.
    These will be used to query the vector store in parallel.
    """
    prompt = """
    You are a knowledgeable academic advisor helping students understand degree
    requirements. For the given question, propose up to five related questions
    that might retrieve useful information from a course catalog or program
    requirements.

    Provide concise, single-topic questions (without compounding sentences)
    that cover different angles of the original question.

    Ensure each question is complete and directly related to the original inquiry.
    List each question on a separate line without numbering.
    """

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": query},
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
    )

    content = response.choices[0].message.content or ""
    # split lines and drop empties
    queries = [line.strip() for line in content.split("\n") if line.strip()]
    return queries

def load_documents_from_directory(directory_path):
    print("==== Loading JSON documents from directory ====")
    try:
        visible = [f for f in os.listdir(directory_path)]
        jsons   = [f for f in visible if f.lower().endswith(".json")]
        print(f"Docs dir: {os.path.abspath(directory_path)}")
        print(f"Found {len(jsons)} JSON files: {jsons}")
    except FileNotFoundError:
        print(f"Documents directory not found: {os.path.abspath(directory_path)}")
        return []
    docs = []
    for filename in os.listdir(directory_path):
        if not filename.lower().endswith(".json"): 
            continue
        path = os.path.join(directory_path, filename)
        try:
            data = json.load(open(path, "r", encoding="utf-8"))
        except Exception as e:
            print(f"Skip {filename}: {e}")
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
    def as_text(x):
        if isinstance(x, dict):
            parts=[]
            term = x.get("term") or x.get("Term") or x.get("semester")
            year = x.get("year") or x.get("year_of_study")
            semester = x.get("semester")

            if term or (semester and year):
                if semester and year:
                    parts.append(
                        f"This corresponds to the {semester} semester of the {year} year "
                        f"(also known as Term {term})."
                    )

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

def query_documents(question, n_results=10, use_multi_query=True):
    # Single-query fallback (if you ever want it)
    if not use_multi_query:
        results = collection.query(query_texts=[question], n_results=n_results)
    else:
        # 1) generate augmented queries
        aug_queries = generate_multi_query(question)

        # 2) combine original + augmented queries
        joint_queries = [question] + aug_queries

        # 3) query Chroma with ALL queries
        results = collection.query(
            query_texts=joint_queries,
            n_results=n_results,
        )

    if not results.get("documents"):
        return []

    seen = set()
    merged_docs = []
    for sub in results["documents"]:
        for doc in sub:
            if doc not in seen:
                seen.add(doc)
                merged_docs.append(doc)

    return merged_docs

def rerank_chunks(question: str, chunks, top_k: int = 6):
    """
    Use the LLM to score which chunks are most relevant to the question.
    Returns a smaller, sorted list of chunks.
    """
    if not chunks:
        return []
    prompt = (
        "You are reranking context passages for a question.\n"
        "Given the question and a list of passages, return the indices of the "
        f"top {top_k} most relevant passages in descending order of relevance.\n"
        "Respond ONLY with a comma-separated list of integers (0-based indices).\n\n"
        f"Question: {question}\n\n"
    )

    for i, ch in enumerate(chunks):
        prompt += f"Passage {i}: {ch}\n\n"

    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0,
        messages=[{"role": "system", "content": prompt}],
    )
    text = resp.choices[0].message.content.strip()
    try:
        indices = [int(x.strip()) for x in text.split(",") if x.strip().isdigit()]
    except Exception:
        return chunks[:top_k]

    ordered = [chunks[i] for i in indices if 0 <= i < len(chunks)]
    return ordered[:top_k]

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
        print("No JSON documents loaded. Skipping index build.")
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
        print("No chunks produced; nothing to add.")
        return

    print("==== Inserting chunks into db ====")
    collection.add(ids=chunked_ids, documents=chunked_texts, metadatas=chunked_metas)
    print("==== Ingest complete ====")




