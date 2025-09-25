from flask import Flask, jsonify, request
import uuid
import os
import json
from dotenv import load_dotenv
from utils.pdf_processor import extract_text_from_pdf, chunk_text
from utils.qdrant_manager import QdrantManager
from utils.llm_wrapper import LLMWrapper

load_dotenv()

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "storage/pdfs"

# make sure folders exist
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs("storage/qdrant", exist_ok=True)

llm = LLMWrapper(use_openai=False)  # set to True for openai False for Gemini
qdrant_mgr = QdrantManager(use_openai=False)  # set to True for openai

PDF_LIMIT = 5
USER_PDF_FILE = "user_pdf.json"

# Load existing user PDF data
if os.path.exists(USER_PDF_FILE):
    with open(USER_PDF_FILE, "r") as f:
        user_pdf = json.load(f)
else:
    user_pdf = {}


@app.route("/")
def home():
    return "PDF Explainer Backend"


@app.route("/upload_pdf", methods=["POST"])
def upload_pdf():
    user_id = request.form.get("user_id") or request.json.get("user_id")
    if not user_id:
        return jsonify(error="Missing user_id"), 400

    if "files" not in request.files:
        return jsonify(error="No files uploaded"), 400

    files = request.files.getlist("files")
    if not files or files[0].filename == "":
        return jsonify(error="No files selected"), 400

    # Initialize user tracking if not exists
    if user_id not in user_pdf:
        user_pdf[user_id] = {"count": 0, "files": []}

    pdf_ids = []
    for uploaded in files:
        # Check for duplicate files by filename
        if any(f["filename"] == uploaded.filename for f in user_pdf[user_id]["files"]):
            return (
                jsonify(
                    error="PDF already exists.",
                    resource={"filename": uploaded.filename},
                ),
                400,
            )
        if user_pdf[user_id]["count"] >= PDF_LIMIT:
            return (
                jsonify(
                    error="Your quota is full. Maximum 5 PDFs allowed.",
                    resource={"count": user_pdf[user_id]["count"]},
                ),
                400,
            )

        pdf_id = str(uuid.uuid4())
        pdf_ids.append(pdf_id)

        # save file
        pdf_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{pdf_id}.pdf")
        uploaded.save(pdf_path)

        # extract & chunk
        full_text = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(full_text)

        # metadata to store alongside each chunk
        metadata = {
            "pdf_id": pdf_id,
            "filename": uploaded.filename,
            "user_id": user_id,
        }

        # store in Qdrant under this user
        qdrant_mgr.add_document(
            collection_name=user_id, chunks=chunks, metadata=metadata
        )

        # Update user tracking for each file
        user_pdf[user_id]["count"] += 1
        user_pdf[user_id]["files"].append(
            {"filename": uploaded.filename, "pdf_id": pdf_id}
        )

    # Save updated user data
    with open(USER_PDF_FILE, "w") as f:
        json.dump(user_pdf, f)

    return (
        jsonify(
            user_id=user_id,
            pdf_ids=pdf_ids,
            message=f"PDF uploaded successfully. {user_pdf[user_id]['count']}/{PDF_LIMIT} PDFs used.",
        ),
        200,
    )


@app.route("/ask_question", methods=["POST"])
def ask_question():
    payload = request.get_json()
    user_id = payload.get("user_id")
    question = payload.get("question")
    pdf_ids = payload.get("pdf_ids")

    if not user_id or not question:
        return jsonify(error="Missing user_id or question"), 400

    context_chunks = qdrant_mgr.query(
        collection_name=user_id, query=question, top_k=10, pdf_ids=pdf_ids
    )
    context = "\n\n".join(context_chunks)
    system_prompt = (
        "You are an assistant that answers questions "
        "based on provided context from selected PDF documents."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

    answer = llm.chat(system_prompt, user_prompt)

    return jsonify(answer=answer), 200


@app.route("/user_status", methods=["GET"])
def user_status():
    user_id = request.args.get("user_id")
    if not user_id:
        return jsonify(error="Missing user_id"), 400

    if user_id not in user_pdf:
        return jsonify(user_id=user_id, count=0, limit=PDF_LIMIT, files=[]), 200

    return (
        jsonify(
            user_id=user_id,
            count=user_pdf[user_id]["count"],
            limit=PDF_LIMIT,
            files=user_pdf[user_id]["files"],
        ),
        200,
    )


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
