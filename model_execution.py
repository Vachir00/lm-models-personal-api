from transformers import pipeline

model_language_detection = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")
model_language_translation = pipeline("translation", model="Helsinki-NLP/opus-mt-es-en")

class ClassXlmRoberta():
    def __init__(self, top_getter: int = 1):
        self.top_getter = top_getter
        self.model = model_language_detection


    def check_language(self, text: str):
        execution = self.model(text, top_k=self.top_getter, truncation=True)
        return execution

class ClassToEnglish():
    def __init__(self):
        self.model = model_language_translation

    def translate_text(self, text: str):
        execution = self.model(text)
        return execution