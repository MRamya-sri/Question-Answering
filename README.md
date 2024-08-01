# Question Answering Flask App

This repository contains a Flask web application that leverages the Hugging Face Transformers library to provide a question-answering service. The application uses the `distilbert-base-cased-distilled-squad` model to answer questions based on a given context.

## Features

- **Question Answering**: Utilizes the `distilbert-base-cased-distilled-squad` model to answer questions.
- **Flask Web Application**: Provides a web interface to input questions and get answers.
- **Concurrent Processing**: Handles requests using concurrent processing to improve performance.

## Prerequisites

- Python 3.7 or higher
- Flask
- Transformers

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/question-answering-flask-app.git
    cd question-answering-flask-app
    ```

2. Create and activate a virtual environment:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

4. Make sure to have the context file `constitution_of_india.json` in the specified path.

## Usage

1. Run the Flask application:
    ```sh
    python app.py
    ```

2. Open your web browser and go to `http://127.0.0.1:5000`.

3. Enter your question in the input field and get the answer.

## File Structure

- `app.py`: The main Flask application file.
- `templates/index.html`: HTML template for the web interface.
- `requirements.txt`: List of required Python packages.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any features, bug fixes, or enhancements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
