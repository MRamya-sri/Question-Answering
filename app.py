from flask import Flask, request, render_template
from transformers import pipeline
from concurrent.futures import ThreadPoolExecutor
import threading

app = Flask(__name__)

model_name = "distilbert-base-cased-distilled-squad"

# Load model & tokenizer (do this once at startup)
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)

# Load context (do this once at startup)
with open('D:\\Question-Answering\\constitution_of_india.json', 'r', encoding='utf-8') as file:
    context = file.read()

def get_answer(question):
    QA_input = {
        'question': question,
        'context': context
    }
    result = nlp(QA_input)
    return result['answer']

@app.route('/')
def answer_question():
    question = request.args.get('question')
    answer = None
    if question:
        # Run the model in a separate thread to avoid blocking
        future = executor.submit(get_answer, question)
        answer = future.result()
    return render_template('index.html', answer=answer)

if __name__ == '__main__':
    app.run(threaded=True)