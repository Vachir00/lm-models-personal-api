from flask import Flask, request
from model_execution import ClassXlmRoberta, ClassToEnglish, ClassToSpanish, ClassTinyLLM
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
        "name": "Qwen/Qwen1.5-4B-Chat"
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
    elif model_selected == "Qwen/Qwen1.5-4B-Chat":
        model = ClassTinyLLM()
        context = data.get("context")
        return model.chat(context, user_text), 200
    else:
        return "Invalid model", 400

@app.route('/list_models', methods=['GET'])
def list_models():  # put application's code here
    return models, 200

if __name__ == '__main__':
    app.run()
