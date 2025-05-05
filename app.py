from flask import Flask, request, render_template
from model import predict_from_input_data

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict_quality_type():
    """Endpoint to predict tomato quality and type."""
    if "images" not in request.files:
        return render_template("error.html", error="No file provided"), 200

    image_file = request.files["images"]

    if (
        image_file
        and image_file.filename
        and image_file.filename.lower().endswith(("png", "jpg", "jpeg"))
    ):
        try:
            result = predict_from_input_data(image_file.read(), threshold=0.85)
            return render_template("result.html", result=result)
        except Exception as e:
            return render_template("error.html", error=str(e)), 200

    return render_template("error.html", error="Invalid file format"), 200


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(port=8080, debug=True)
