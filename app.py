# from flask import Flask, request, render_template
# from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer
# import os

# app = Flask(__name__)

# # Check if fine-tuned model exists, otherwise use the base model
# if os.path.exists("./fine_tuned_qa_model"):
#     model_path = "./fine_tuned_qa_model"
# else:
#     model_path = "distilbert-base-cased-distilled-squad"
#     print("Fine-tuned model not found. Using base model.")

# # Load the model and tokenizer
# model = AutoModelForQuestionAnswering.from_pretrained(model_path)
# tokenizer = AutoTokenizer.from_pretrained(model_path)

# # Load the context from the file
# try:
#     with open(r'D:/Question-Answering/context.txt', 'r', encoding='utf-8') as file:
#         context = file.read()
# except FileNotFoundError:
#     print("Context file not found. Please ensure 'context.txt' is in the correct location.")
#     context = ""

# # Initialize the question-answering pipeline
# nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)

# @app.route('/')
# def answer_question():
#     # retrieve the question entered by the user
#     question = request.args.get('question')

#     answer = None
#     if question and context:
#         try:
#             QA_input = {
#                 'question': question,
#                 'context': context
#             }
#             res = nlp(QA_input)
#             answer = res['answer']
#         except Exception as e:
#             print(f"An error occurred: {str(e)}")
#             answer = "Sorry, I couldn't process that question. Please try again."
#     elif not context:
#         answer = "Sorry, the context is not available. Please check the server configuration."

#     # render the index.html template and pass the answer to it as a variable
#     return render_template('index.html', answer=answer)

# if __name__ == '__main__':
#     app.run(debug=True)