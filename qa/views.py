import json
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

# import your RAG logic from app.py
from app import build_index_if_missing, query_documents, generate_response, rerank_chunks

# Build index once when the server starts (no-op if already built)
build_index_if_missing("./documents")

@csrf_exempt  # keeps things simple for now (no CSRF token handling in JS)
def home(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body.decode("utf-8"))
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON"}, status=400)

        question = data.get("question", "").strip()
        if not question:
            return JsonResponse({"error": "Empty question"}, status=400)

        chunks = query_documents(question)
        if not chunks:
            answer = (
                "I couldn't find any relevant information in the catalog "
                "for that question. Try rephrasing or being more specific.")
        else:
            best_chunks = rerank_chunks(question, chunks, top_k=12)
            answer = generate_response(question, best_chunks)
        

        return JsonResponse({"answer": answer})

    # GET → render the single landing page
    return render(request, "qa/home.html")
