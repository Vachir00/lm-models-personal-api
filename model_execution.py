import torch
from transformers import pipeline

model_language_detection = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")
model_toenglish_translation = pipeline("translation", model="Helsinki-NLP/opus-mt-es-en")
model_tospanish_translation = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es")
model_tiny_llm = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")

class ClassXlmRoberta():
    def __init__(self, top_getter: int = 1):
        self.top_getter = top_getter
        self.model = model_language_detection


    def check_language(self, text: str):
        execution = self.model(text, top_k=self.top_getter, truncation=True)
        return execution

class ClassToEnglish():
    def __init__(self):
        self.model = model_toenglish_translation

    def translate_text(self, text: str):
        execution = self.model(text)
        return execution

class ClassToSpanish():
    def __init__(self):
        self.model = model_tospanish_translation

    def translate_text(self, text: str):
        execution = self.model(text)
        return execution

class ClassTinyLlama():
    def __init__(self):
        self.model = model_tiny_llm

    def chat(self, context: str, text: str):
        payload = [
            {
                "role": "system",
                "content": context,
            },
            {
                "role": "user",
                "content": text
            }
        ]
        prompt = self.model.tokenizer.apply_chat_template(payload, tokenize=False, add_generation_prompt=True)
        execution = self.model(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
        delimiter = "<|assistant|>"
        separate_text = execution[0].get("generated_text").split(delimiter)
        assistant_response = separate_text[1].strip()
        return assistant_response