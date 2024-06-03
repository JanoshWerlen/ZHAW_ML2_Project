import json
from pathlib import Path
import numpy as np
import openai
import sqlite3
from getpass import getpass
from openai import OpenAI
import faiss
import os
from dotenv import load_dotenv

# Configure OpenAI API key
client = OpenAI()
load_dotenv()

client.api_key = os.getenv('OPENAI_API_KEY')


model_large = "text-embedding-3-large"
model_small = "text-embedding-3-small"
# model = "text-embedding-3-small"
index_path_ABPR = "data/ABPR/articles_large.index"
json_path_ABPR = "data/ABPR/articles_large.json"
index_path_ABPR_small = "data/ABPR/articles.index"
json_path_ABPR_small = "data/ABPR/articles.json"

index_path_ARG = "data/ARG/articles.index"
json_path_ARG = "data/ARG/articles_embedded.json"

index_path_KAR = "data/KAR/articles.index"
json_path_KAR = "data/KAR/articles_embedded.json"
# index_path = "data/ARG/articles_incl_ARG.index"
# json_path = "data/ARG/articles_incl_ARG_embedded.json"

# Database setup
conn = sqlite3.connect('chatbot.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE,
                password TEXT)''')

c.execute('''CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                message TEXT,
                response TEXT,
                action TEXT,
                articles TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id))''')

conn.commit()

# User authentication functions


def register(username, password):
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)",
                  (username, password))
        conn.commit()
        print("Registration successful!")
    except sqlite3.IntegrityError:
        print("Username already exists. Please choose a different username.")


def login(username, password):
    c.execute("SELECT * FROM users WHERE username = ? AND password = ?",
              (username, password))
    user = c.fetchone()
    if user:
        print("Login successful!")
        return user[0]  # Return user_id
    else:
        print("Invalid username or password.")
        return None


def get_embedding_large(text, tags, model=model_large):
    text = text.replace("\n", " ")
    combine = text + " " .join(tags)
    return openai.embeddings.create(input=[combine], model=model).data[0].embedding


def get_embedding_small(text, tags, model=model_small):
    text = text.replace("\n", " ")
    combine = text + " " .join(tags)
    return openai.embeddings.create(input=[combine], model=model).data[0].embedding

# Function to load JSON data from a file


def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

# Function to save JSON data to a file
# data = load_json(json_path)


def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def refine_query(query):
    print("\nOriginales Query: " + query + "\n")
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"Erweitere die User Frage: <anfrage>{query}</anfrage>, mit Suchbegriffen, damit die Frage möglichst gute RAG ergebnisse liefert. Der erweiterte Anfrage soll ich immer auf den Kontext einer Anstellung am Stadtspital Zürich in der Schweiz beziehen. Alle Antworten sollen als Fragen formuliert sein"},
        ]
    )
    response = response.choices[0].message.content
    print("\nimproving query... \n")
    print("\nimproved query: " + response + "\n")
    return response


def generate_query_embedding(query, filtered_values):
    if filtered_values in ["ABPR", "KAR", "ARG"]:
        return get_embedding_large(query, [])
    elif filtered_values == "ARG":
        return get_embedding_small(query, [])
    else:
        raise ValueError("Invalid filtered_values provided")

def get_files(filter_values):

    if filter_values == "ABPR":
        print("Filter ABPR")
        index = faiss.read_index(index_path_ABPR)
        data = load_json(json_path_ABPR)
        context = "nichtärztliche"
    if filter_values == "ARG":
        print("Filter ARG")
        index = faiss.read_index(index_path_ARG)
        data = load_json(json_path_ARG)
        context = "Arbeitsgesetz"
    if filter_values == "KAR":
        print("Filter KAR")
        index = faiss.read_index(index_path_KAR)
        data = load_json(json_path_KAR) 
        context = "Ärztliches Person, Oberärzte, Kaderärzte, Ärzte"      

    return index, data, context


def get_response_string(refind_query, filter_values):
    print("\nFinding relevant articles... \n")
    index, data , context = get_files(filter_values)

    refind_query =  refind_query + " betreffend " + context

    query_embedding = np.array(generate_query_embedding(
        refind_query, filter_values)).astype('float32').reshape(1, -1)

    # Perform the search
    k = 5  # Number of nearest neighbors to retrieve
    distances, indices = index.search(query_embedding, k)

    # Ensure 'indices' is a list of integers and not out of range
    indices = indices[0]

    # Retrieve the matching articles
    matching_articles = [data[i]
                         # Direct access by index
                         for i in indices if i < len(data)]
    response_string = ""
    # Print the matching articles
    for article in matching_articles:
        response_string += f"Artikel: {article['article_number']}, Gesetzestext: {article['metadata'].get('Gesetzestext')}, Title: {article['title']},\n Text: {article['text']}\n"
    print("\nRelevant Articles found... \n")    
    print("\nRelevant Articles: " + response_string + "\n")
    
    return response_string, filter_values

def perform_rag_request(message, filter_values, additional_context=""):
     
    combined_query = f"{additional_context} {message}".strip()

    refind_query = refine_query(combined_query)

    print("\n performing RAG based on query: " + str(refind_query) + "\n")

    # Combine the refined query with additional context if provided
   

    response_string, filter = get_response_string(combined_query, filter_values)

    response_string, filtered_articles = check_rag_for_context(response_string, filter)

    system_query = f"""Du bist ein HR-Assistent des Stadtspitals Zürich, welcher Fragen von Angestellten beantwortet. Antworte basierend auf den Inhalten in den folgenden Artikeln: <Artikelinhalt>{response_string}</artikelinhalt>.
    Antworte professionell und kurz ohne Begrüssung oder Verabschiedung. Verwende direkte Zitate aus den Artikeln und setze diese in Anführungszeichen. Gib am Ende eine Liste aller relevanten Artikel und Artikeltitel an. Bei Fragen welche überhaupt nichts mit der Arbeit zutun haben, lenke den zurück zum Thema. """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"{system_query}"},
            {"role": "user", "content": f"{combined_query}"}
        ]
    )
    response = response.choices[0].message.content
    return response, filtered_articles



inquired_infos = []
times_inquired = 0


def inquire_more_information(message):
    global times_inquired
    times_inquired += 1
    print(f"\n currently inquired {times_inquired} times... \n")

    print("Aktuelle convo...: " + str(inquired_infos))
    print("\n Inquiring more info... \n")
    system_query = f"Frage genauer nach, was der User wissen will um die originale Anfrage '{
        message}' zu präzisieren. Akzeptiere nur personalrechtliche Fragen welche um in deiner Rolle als HR-Berater am Stadtspital Zürich relevant sind. Geh davon aus, dass der User immer ein Angestellter des Stadtspitals Zürich ist."
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



def check_rag_for_context(message, filter):
    print(f"\n checking for context... \n")

    system_query = f"Suche im folgenden text nach allen genannten Artikeln und retourniere ausschliesslich eine Liste im format ['Art. X', 'Art. Y>'] der gefunden Atrikel: <text> {message} </text>, Ignoriere Allen Text vor dem 'Art.' und alles nach der Zahl."
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"{system_query}"},
        ],
    )
    response = response.choices[0].message.content

    print("\nGeefundene Artikel: ")
    response = response.strip().rstrip('>').replace('>','')
    print(response)

    # Parse the response to extract the list of article numbers
    try:
        article_numbers = eval(response)
    except SyntaxError:
        print("Error: Invalid syntax in response. Please check the response format.")
        return "", ""

    if filter == "ABPR":
        json_file = Path('data/ABPR/articles_large.json')
    elif filter =="KAR":
        json_file = Path('data/KAR/articles_embedded.json')
    elif filter =="ARG":
        json_file = Path('data/ARG/articles.json')    

    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    response_string = ""    
    articles = ""

    filtered_articles = [article for article in data if article['article_number'] in article_numbers]

    for article in filtered_articles:
        response_string += f"Article Number: {article['article_number']}, Gesetzestext: {article['metadata'].get('Gesetzestext')}, Title: {article['title']},\n Text: {article['text']}\n"
        articles += f"{article['article_number']},"

    print("\nGefilterte Texte: ")
    print(response_string)

    

    return response_string, articles

# Load the JSON file


# Filter the JSON data for matching article numbers


# Print the filtered articles



def decide_action(message):
    print("\n deciding... \n")
    prompt = f"Given the following user message, decide what action should be taken. The options are: perform_rag_request, inquire_more_information, end_conversation. If the User asks a relevant question regarding employment or similar, decide to 'perform_rag_request'. End the conversation if the Questions are mean, unprofessional or insulting.  \n\nUser message: {message}\n\nAction:"
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"{prompt}"},
        ],
        max_tokens=10,
        stop=['\n']
    )

    action = response.choices[0].message.content
    print(f"\n Getroffene Entscheidung: {action} \n")
    return action

def get_user_input():
    message = input("> ")
    return message

def decide_continue():
    print("\nDeciding Followup")
    while True:
        print("\nWillst du eine Folgefrage stellen? (Ja, Nein)\n")
        user_input = get_user_input().strip().lower()
        prompt = f"Given the following user message, decide what action should be taken. The options are: followup, followup_with_question, newquestion. Choose 'followup' if the user-response indicates a positive sentiment to the Question 'Willst du eine Folgefrage stellen?', If the user response contains already a question, decide 'followup_with_question'. Choose 'newquestion' if the user does not want to ask a followup question. \n\nUser message: {user_input}\n\nAction:"
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"{prompt}"},
            ],
            max_tokens=10,
            stop=['\n']
        )
        print("\nFollowup decision: ")
        print(response.choices[0].message.content)
        return response.choices[0].message.content, user_input

        
def start_convo():
    print("\nFrage Eingeben: ")
    message = get_user_input()

    if message.lower() == 'quit':
        return "quit", ""
    
    print("Welche Mitarbeitergruppe betirfft deine Frage?")
    print("1: Nicht-Ärzliches Personal")
    print("2: Assistenzärzte")
    print("3: Kaderärzte (OA.i.V. mit FA / OA / LA / CA)")
    print("4: Unklar / Keine Angabe\n")
    choice = get_user_input()
    if choice == "1":
        filter_values = "ABPR"
    elif choice == "2":
        filter_values = "ARG"
    elif choice == "3":
        filter_values = "KAR"
    else:
        filter_values = ["ABPR", "ARG","KAR"]  # Corrected assignment
    print("\n Filter chosen: " + str(filter_values) + "\n")
    return message, filter_values

# Chatbot function
def user_interaction(user_id, inquired_infos):
    global times_inquired
    keepalive = True
    while keepalive:
        message, filter_values = start_convo()
        if message.lower() == 'quit':
            break
        inquired_infos = [message]  # Reset inquired_infos to only include the latest question
        additional_context = ""  # Initialize additional context
        times_inquired = 0
        while True:
            action = decide_action(str(inquired_infos))
            if times_inquired > 3:
                reply, filtered_articles = perform_rag_request(str(inquired_infos[-1]), filter_values, additional_context)
                break
            elif action == "end_conversation":
                keepalive = False
                print("Bot: Ending the conversation.")
                break  # Exit the loop if conversation should end
            elif action == "perform_rag_request":
                reply, filtered_articles = perform_rag_request(str(inquired_infos[-1]), filter_values, additional_context)
                print("Bot:", reply)
                followup_action, user_input = decide_continue()
                if followup_action == "followup":
                    print("Bitte gebe deine Folgefrage ein:")
                    followup_message = input("You: ").strip()
                    inquired_infos.append(followup_message)
                    print("\nNew info for followup\n")
                    additional_context = reply  # Update additional context with the latest RAG response
                    continue  # Continue the inner loop for follow-up questions
                elif followup_action == "followup_with_question":
                    inquired_infos.append(user_input)
                    print("\nNew info for followup\n")
                    additional_context = reply  # Update additional context with the latest RAG response
                elif followup_action == "newquestion":
                    break  # Exit the inner loop to start a new question
            elif action == "inquire_more_information":
                reply = inquire_more_information(str(inquired_infos[-1]))
                print("Bot:", reply)
                message = input("You: ").strip()
                inquired_infos.append(message)
                if message.lower() == 'quit':
                    return  # Exit the chatbot if user decides to quit

        # Log interaction
        c.execute("INSERT INTO logs (user_id, message, response, articles, action) VALUES (?, ?, ?, ?, ?)",
                  (user_id, message, reply, filtered_articles, action))
        conn.commit()



def chatbot(user_id):
    inquired_infos = []
    print("Chatbot initialized. Type 'quit' to exit.")
    user_interaction(user_id, inquired_infos)


def main():
    print("Welcome! Please choose an option:")
    print("1. Register")
    print("2. Login")
    choice = input("Enter choice: ")

    username = input("Username: ")
    password = getpass("Password: ")

    if choice == '1':
        register(username, password)
        user_id = login(username, password)
        if user_id:
            chatbot(user_id)
    elif choice == '2':
        user_id = login(username, password)
        if user_id:
            chatbot(user_id)
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()
