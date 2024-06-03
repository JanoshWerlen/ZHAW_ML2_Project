import json
from pathlib import Path
import numpy as np
import openai
import faiss
import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template, g
from database import *
from getpass import getpass

# Configure OpenAI API key
client = openai.OpenAI()
load_dotenv()

client.api_key = os.getenv('OPENAI_API_KEY')

# Load Models
model_large = "text-embedding-3-large"
model_small = "text-embedding-3-small"

index_path_ABPR = "data/ABPR/articles_large.index"
json_path_ABPR = "data/ABPR/articles_large.json"
index_path_ABPR_small = "data/ABPR/articles.index"
json_path_ABPR_small = "data/ABPR/articles.json"

index_path_ARG = "data/ARG/articles.index"
json_path_ARG = "data/ARG/articles_embedded.json"

index_path_KAR = "data/KAR/articles.index"
json_path_KAR = "data/KAR/articles_embedded.json"

# Database setup
c, conn = setup_database()

inquired_infos = []
times_inquired = 0

# Load JSON file
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

# Your other functions
def get_embedding_large(text, tags, model=model_large):
    text = text.replace("\n", " ")
    combine = text + " " .join(tags)
    return openai.embeddings.create(input=[combine], model=model).data[0].embedding

def get_embedding_small(text, tags, model=model_small):
    text = text.replace("\n", " ")
    combine = text + " " .join(tags)
    return openai.embeddings.create(input=[combine], model=model).data[0].embedding

def generate_query_embedding(query, filtered_values):
    if filtered_values in ["ABPR", "KAR", "ARG"]:
        return get_embedding_large(query, [])
    elif filtered_values == "ARG":
        return get_embedding_small(query, [])
    else:
        raise ValueError("Invalid filtered_values provided")

def get_files(filter_values):
    if filter_values == "ABPR":
        index = faiss.read_index(index_path_ABPR)
        data = load_json(json_path_ABPR)
        context = "nichtärztliche"
    if filter_values == "ARG":
        index = faiss.read_index(index_path_ARG)
        data = load_json(json_path_ARG)
        context = "Arbeitsgesetz"
    if filter_values == "KAR":
        index = faiss.read_index(index_path_KAR)
        data = load_json(json_path_KAR)
        context = "Ärztliches Person, Oberärzte, Kaderärzte, Ärzte"
    return index, data, context

def refine_query(query):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"Erweitere die User Frage: <anfrage>{query}</anfrage>, mit Suchbegriffen, damit die Frage möglichst gute RAG ergebnisse liefert. Der erweiterte Anfrage soll ich immer auf den Kontext einer Anstellung am Stadtspital Zürich in der Schweiz beziehen. Alle Antworten sollen als Fragen formuliert sein"},
        ]
    )
    response = response.choices[0].message.content
    return response

def get_rag_string(refind_query, filter_values):
    index, data, context = get_files(filter_values)
    refind_query = refind_query + " betreffend " + context

    query_embedding = np.array(generate_query_embedding(
        refind_query, filter_values)).astype('float32').reshape(1, -1)

    k = 5
    distances, indices = index.search(query_embedding, k)
    indices = indices[0]

    matching_articles = [data[i]
                         for i in indices if i < len(data)]
    response_string = ""

    for article in matching_articles:
        response_string += f"Artikel: {article['article_number']}, Gesetzestext: {article['metadata'].get('Gesetzestext')}, Title: {article['title']},\n Text: {article['text']}\n"

    return response_string, filter_values

def check_rag_for_context(message, filter):
    system_query = f"Suche im folgenden text nach allen genannten Artikeln und retourniere ausschliesslich eine Liste im format ['Art. X', 'Art. Y>'] der gefunden Atrikel: <text> {message} </text>, Ignoriere Allen Text vor dem 'Art.' und alles nach der Zahl."
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"{system_query}"},
        ],
    )
    response = response.choices[0].message.content

    response = response.strip().rstrip('>').replace('>', '')

    try:
        article_numbers = eval(response)
    except SyntaxError:
        return "", ""

    if filter == "ABPR":
        json_file = Path('data/ABPR/articles_large.json')
    elif filter == "KAR":
        json_file = Path('data/KAR/articles_embedded.json')
    elif filter == "ARG":
        json_file = Path('data/ARG/articles.json')

    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    response_string = ""
    articles = ""

    filtered_articles = [article for article in data if article['article_number'] in article_numbers]

    for article in filtered_articles:
        response_string += f"Article Number: {article['article_number']}, Gesetzestext: {article['metadata'].get('Gesetzestext')}, Title: {article['title']},\n Text: {article['text']}\n"
        articles += f"{article['article_number']},"

    return response_string, articles

def perform_rag_request(message, filter_values, additional_context=""):
    combined_query = f"{additional_context} {message}".strip()
    refind_query = refine_query(combined_query)

    response_string, filter = get_rag_string(combined_query, filter_values)
    response_string, filtered_articles = check_rag_for_context(response_string, filter)

    system_query = f"""Du bist ein HR-Assistent des Stadtspitals Zürich, welcher Fragen von Angestellten beantwortet. Antworte basierend auf den Inhalten in den folgenden Artikeln: <Artikelinhalt>{response_string}</artikelinhalt> und nur wenn die Inhalte relevant zur Frage sind.
    Antworte professionell und kurz ohne Begrüssung oder Verabschiedung. Verwende direkte Zitate aus den Artikeln und setze diese in Anführungszeichen. Formatiere deine Antwort, dass diese gut lesbar ist und unterteile die Inhalte in Abschnitte. Gib am Ende eine Liste aller relevanten Artikel und Artikeltitel an. Bei Fragen welche überhaupt nichts mit der Arbeit zutun haben, lenke den User zurück zum Thema. """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": f"{system_query}"},
            {"role": "user", "content": f"{refind_query}"}
        ]
    )
    response = response.choices[0].message.content
    return response, filtered_articles

def inquire_more_information(message):
    global times_inquired
    times_inquired += 1
    system_query = f"Frage genauer nach, was der User wissen will um die originale Anfrage '{message}' zu präzisieren. Akzeptiere nur personalrechtliche Fragen welche um in deiner Rolle als HR-Berater am Stadtspital Zürich relevant sind. Geh davon aus, dass der User immer ein Angestellter des Stadtspitals Zürich ist."
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"{system_query}"},
        ],
        max_tokens=100,
        stop=['\n']
    )
    response = response.choices[0].message.content
    return response

def decide_action(message, type):
    if type == 1:
        prompt = f"Given the following user message, decide what action should be taken. The options are: perform_rag_request, inquire_more_information, end_conversation. If the User asks a relevant question regarding employment or similar, decide to 'perform_rag_request'. End the conversation if the Questions are mean, unprofessional or insulting.  \n\nUser message: {message}\n\nAction:"
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"{prompt}"},
            ],
            max_tokens=10,
            stop=['\n']
        )
        action = response.choices[0].message.content.strip()
        return action, message

    elif type == 2:
        while True:
            prompt = f"Given the following user message, decide what action should be taken. The options are: followup, followup_with_question, newquestion. Choose 'followup' if the user-response indicates a positive sentiment to the Question 'Willst du eine Folgefrage stellen?', If the user response contains already a question, decide 'followup_with_question'. Choose 'newquestion' if the user does not want to ask a followup question. \n\nUser message: {message}\n\nAction:"
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": f"{prompt}"},
                ],
                max_tokens=10,
                stop=['\n']
            )
            action = response.choices[0].message.content.strip()
            if action:
                return action, message

def initiate_conversation(user_message, filter_values):
    
    action, message = decide_action(user_message, type=1)


    if action == "perform_rag_request":
        response, articles = perform_rag_request(user_message, filter_values, additional_context="")
    elif times_inquired >= 3:
        response = "The conversation has been ended due to the nature of the inquiry."
    elif action == "inquire_more_information":
        response = inquire_more_information(user_message)
    elif action == "end_conversation":
        response = "Thank you for your message. The conversation has been ended due to the nature of the inquiry."

    return response

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get("message")
    filter_values = data.get("filter_values")
    response = initiate_conversation(user_message, filter_values)
    return jsonify({"response": response})


@app.route('/login', methods=['POST'])
def login_user():
    data = request.json
    username = data.get("username")
    password = data.get("password")
    user_id = login(username, password, c)
    if user_id:
        return jsonify({"user_id": user_id})
    else:
        return jsonify({"error": "Invalid credentials"}), 401

@app.route('/register', methods=['POST'])
def register_user():
    data = request.json
    username = data.get("username")
    password = data.get("password")
    register(username, password, c, conn)
    user_id = login(username, password, c)
    if user_id:
        return jsonify({"user_id": user_id})
    else:
        return jsonify({"error": "Registration failed"}), 500

if __name__ == "__main__":
    app.run(debug=True)
