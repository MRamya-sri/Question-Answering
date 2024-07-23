import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
import json

# Load the model and tokenizer
model_name = "distilbert-base-cased-distilled-squad"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the context
with open(r'D:/Question-Answering/context.txt', 'r', encoding='utf-8') as file:
    context = file.read()

# Prepare the dataset
qa_pairs = [
    {
        "question": "What is dopamine and how does it affect motivation?",
        "answer": "Dopamine is a neurotransmitter in the brain that makes us desire things, providing motivation to act. It plays a crucial role in our reward system, influencing behavior by making us anticipate rewards from certain activities."
    },
    {
        "question": "How does excessive exposure to high-dopamine activities affect our behavior?",
        "answer": "Excessive exposure to high-dopamine activities can lead to dopamine tolerance. This means that activities that once provided satisfaction now require more stimulation to achieve the same effect, making lower-dopamine activities less appealing."
    },
    {
        "question": "What is a dopamine detox and how is it performed?",
        "answer": "A dopamine detox is a practice aimed at resetting our dopamine sensitivity by temporarily avoiding high-stimulation activities. It involves setting aside time to avoid highly stimulating activities and embracing simpler, less stimulating tasks."
    },
    {
        "question": "How can dopamine detox principles be incorporated into daily life?",
        "answer": "Dopamine detox principles can be incorporated into daily life by choosing specific times to abstain from high-dopamine behaviors, prioritizing low-dopamine activities earlier in the day, and using a reward system to balance low and high dopamine activities."
    }
]

# Prepare the dataset
train_data = []
for pair in qa_pairs:
    train_data.append({
        "context": context,
        "question": pair["question"],
        "answers": {
            "text": [pair["answer"]],
            "answer_start": [context.index(pair["answer"])]
        }
    })

# Convert to Dataset object
dataset = Dataset.from_list(train_data)

# Tokenize the dataset
def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label is (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

tokenized_dataset = dataset.map(preprocess_function, remove_columns=dataset.column_names)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./fine_tuned_qa_model")
tokenizer.save_pretrained("./fine_tuned_qa_model")

print("Fine-tuning complete. Model saved to ./fine_tuned_qa_model")