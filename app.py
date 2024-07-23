from flask import Flask, request, render_template
from transformers import pipeline

app = Flask(__name__)

model_name = "distilbert-base-cased-distilled-squad"

# Load the context from the file
with open(r'', 'r', encoding='utf-8') as file:
    context = file.read()

# Initialize the question-answering pipeline
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)

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
    app.run(debug=True)