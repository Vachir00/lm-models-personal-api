import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

model_language_detection = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")
model_toenglish_translation = pipeline("translation", model="Helsinki-NLP/opus-mt-es-en")
model_tospanish_translation = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es")
model_tiny_llm = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-4B-Chat", torch_dtype="auto", device_map="auto")

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

class ClassTinyLLM():
    def __init__(self):
        self.model = model_tiny_llm
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-4B-Chat")

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
        text = self.tokenizer.apply_chat_template(payload, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to("cpu")
        generated_ids = self.model.generate(model_inputs.input_ids, max_new_tokens=256)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response