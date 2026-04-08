import os
from flask import Flask, request, render_template, redirect, flash
from werkzeug.utils import secure_filename

import esg_analyzer

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"pdf"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["SECRET_KEY"] = "a_secret_key_for_flash_messages"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def upload_and_process():
    if request.method == "POST":
        # Handle upload
        if "file" not in request.files:
            flash("請求中沒有檔案部分")
            return redirect(request.url)

        file = request.files["file"]

        if file.filename == "":
            flash("沒有選取檔案")
            return redirect(request.url)

        if not (file and allowed_file(file.filename)):
            flash("檔案格式不正確，請上傳 PDF")
            return redirect(request.url)

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # Form fields
        company = request.form.get("company", "未指定公司")
        year = request.form.get("year", "2025")

        # Query: prefer dropdown "topic" (碳排/資安/水資源), fallback to old "query"
        topic = request.form.get("topic", "").strip()
        if not topic:
            topic = request.form.get("query", "資安").strip()

        # Validate topic to keep it aligned with TOPIC2GRI keys
        valid_topics = getattr(esg_analyzer, "TOPIC2GRI", {}).keys()
        if topic not in valid_topics:
            # allow user to type other keywords, but if you want to restrict strictly, uncomment next two lines:
            # flash("查詢主題不在允許清單中，請改用下拉選單")
            # return redirect(request.url)
            pass

        # Optional: disable export during web requests to avoid generating many files
        export_all_chunks = request.form.get("export_all_chunks") == "1"

        try:
            results = esg_analyzer.run_pipeline(
                pdf_paths=[filepath],
                company=company,
                year=year,
                query=topic,
                topk=12,
                export_all_chunks=False
            )

            # Pass options back to templates for dropdown and sticky selection
            topic_options = getattr(esg_analyzer, "TOPIC_OPTIONS", ["碳排", "資安", "水資源"])

            return render_template(
                "results.html",
                results=results,
                filename=filename,
                company=company,
                year=year,
                selected_topic=topic,
                topic_options=topic_options,
                export_all_chunks=False
            )

        except Exception as e:
            flash(f"分析檔案時發生錯誤: {e}")
            return redirect(request.url)

    # GET request: show upload page with dropdown options
    topic_options = getattr(esg_analyzer, "TOPIC_OPTIONS", ["碳排", "資安", "水資源"])
    return render_template("index.html", topic_options=topic_options, selected_topic="資安")


if __name__ == "__main__":
    # Disable reloader to avoid running analysis twice in debug mode
    app.run(debug=True, port=5000, use_reloader=False)