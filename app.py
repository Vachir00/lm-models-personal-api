from flask import Flask, request
from model_execution import ClassXlmRoberta, ClassToEnglish, ClassToSpanish, ClassTinyLlama
app = Flask(__name__)

# MODELS
models = [
    {
        "type": "language_detection",
        "name": "papluca/xlm-roberta-base-language-detection"
    },
    {
        "type": "language_translation",
        "name": "Helsinki-NLP/opus-mt-es-en"
    },
    {
        "type": "language_translation",
        "name": "Helsinki-NLP/opus-mt-en-es"
    },
    {
        "type": "chatbot",
        "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    }
]

@app.route('/execute_model', methods=['POST'])
def execute_model():
    data = request.get_json()
    model_selected = data.get("model")
    if not model_selected in [model['name'] for model in models]:
        return "Invalid model1", 400

    user_text = data.get("text")
    if model_selected == "papluca/xlm-roberta-base-language-detection":
        model = ClassXlmRoberta()
        return model.check_language(user_text), 200
    elif model_selected == "Helsinki-NLP/opus-mt-es-en":
        model = ClassToEnglish()
        return model.translate_text(user_text), 200
    elif model_selected == "Helsinki-NLP/opus-mt-en-es":
        model = ClassToSpanish()
        return model.translate_text(user_text), 200
    elif model_selected == "TinyLlama/TinyLlama-1.1B-Chat-v1.0":
        model = ClassTinyLlama()
        context = data.get("context")
        return model.chat(context, user_text), 200
    else:
        return "Invalid model", 400

@app.route('/list_models', methods=['GET'])
def list_models():  # put application's code here
    return models, 200

if __name__ == '__main__':
    app.run()
