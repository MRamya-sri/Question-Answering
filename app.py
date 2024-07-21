from flask import Flask, request, render_template
from transformers import pipeline

app = Flask(__name__)

model_name = "distilbert-base-cased-distilled-squad"

# a) Get predictions
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)

# b) Load model & tokenizer
model = model_name

with open('D:\Hugging_Face\Question-Answering\context.txt', 'r', encoding='utf-8') as file:
    context = file.read()

@app.route('/')
def answer_question():
    # retrieve the question entered by the user
    question = request.args.get('question')

    answer = None
    if question:
        QA_input = {
            'question': question,
            'context': context
        }
        res = nlp(QA_input)
        answer = res['answer']

    # render the index.html template and pass the answer to it as a variable
    return render_template('index.html', answer=answer)

if __name__ == '__main__':
    app.run()
