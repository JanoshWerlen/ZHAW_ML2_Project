import json
from pathlib import Path
import numpy as np
import openai

from getpass import getpass
from openai import OpenAI
import faiss
import os
from dotenv import load_dotenv

from database import *

# Configure OpenAI API key
client = OpenAI()
load_dotenv()

client.api_key = os.getenv('OPENAI_API_KEY')

#Load Models
model_large = "text-embedding-3-large"
model_small = "text-embedding-3-small"

index_path_ABPR = "data/JSON/articles_main.index"
json_path_ABPR = "data/JSON/articles_main_embedded.json"
#index_path_ABPR = "data/ABPR/articles_large.index"
#json_path_ABPR = "data/ABPR/articles_large.json"
#index_path_ABPR_small = "data/ABPR/articles.index"
#json_path_ABPR_small = "data/ABPR/articles.json"

index_path_ARG = "data/ARG/articles.index"
json_path_ARG = "data/ARG/articles_embedded.json"

index_path_KAR = "data/KAR/articles.index"
json_path_KAR = "data/KAR/articles_embedded.json"


# Database setup
c, conn = setup_database()


inquired_infos = []
times_inquired = 0


#load JSON file
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def bot_response(message):
    print("\nBot: " )
    print(message)

def get_user_input():
    print("Bitte gib eine Frage ein:\n")
    message = input("\n> ")
    return message    

#Get embeddings depending on model type
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
    

#Get files depending on filter
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


def get_rag_string(refind_query, filter_values):
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
        json_file = Path('data/JSON/articles_main_embedded.json')
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

def perform_rag_request_with_context(message, filter_values, additional_context=""):
    combined_query = f"{additional_context} {message}".strip()
    refind_query = refine_query(combined_query)

    print("\n performing RAG based on query: " + str(refind_query) + "\n")

    # Combine the refined query with additional context if provided
   
    response_string, filter = get_rag_string(combined_query, filter_values)

    response_string, filtered_articles = check_rag_for_context(response_string, filter)

    system_query = f"""Du bist ein HR-Assistent des Stadtspitals Zürich, welcher Fragen von Angestellten beantwortet. Antworte basierend auf den Inhalten in den folgenden Artikeln: <Artikelinhalt>{response_string}</artikelinhalt> und nur wenn die Inhalte relevant zur Frage sind.
    Antworte professionell und kurz ohne Begrüssung oder Verabschiedung. Verwende direkte Zitate aus den Artikeln und setze diese in Anführungszeichen. Gib am Ende eine Liste aller relevanten Artikel und Artikeltitel an. Bei Fragen welche überhaupt nichts mit der Arbeit zutun haben, lenke den User zurück zum Thema. """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": f"{system_query}"},
            {"role": "user", "content": f"{refind_query}"}
        ]
    )
    response = response.choices[0].message.content
    return response, filtered_articles


def perform_rag_request(message, filter_values, additional_context=""):
    combined_query = f"{additional_context} {message}".strip()
    refind_query = refine_query(combined_query)

    print("\n performing RAG based on query: " + str(refind_query) + "\n")

    # Combine the refined query with additional context if provided
   
    response_string, filter = get_rag_string(combined_query, filter_values)

    system_query = f"""Du bist ein HR-Assistent des Stadtspitals Zürich, welcher Fragen von Angestellten beantwortet. Antworte basierend auf den Inhalten in den folgenden Artikeln: <Artikelinhalt>{response_string}</artikelinhalt> und nur wenn die Inhalte relevant zur Frage sind.
    Antworte professionell und kurz ohne Begrüssung oder Verabschiedung. Verwende direkte Zitate aus den Artikeln und setze diese in Anführungszeichen. Gib am Ende eine Liste aller relevanten Artikel und Artikeltitel an. Bei Fragen welche überhaupt nichts mit der Arbeit zutun haben, lenke den User zurück zum Thema. """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": f"{system_query}"},
            {"role": "user", "content": f"{refind_query}"}
        ]
    )
    response = response.choices[0].message.content
    return response




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
    bot_response(response)
    user_input = get_user_input()
    return user_input






def decide_action(message, type):

    if type == 1:
        print("\n evaluating Question... \n")
        prompt = f"Given the following user message, decide what action should be taken. The options are: perform_rag_request, inquire_more_information, end_conversation. If the User asks a relevant question regarding the employment or thing relevant to the employment, decide to 'perform_rag_request'. End the conversation if the Questions are mean, unprofessional or insulting. Inquire more Information if the Question is unclear.  \n\nUser message: {message}\n\nAction:"
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
        return action, message
    
    elif type == 2:
        print("\nDeciding on Followup")
        while True:
            bot_response("\nWillst du eine Folgefrage stellen? (Folgefragen versuchen den bisherigen Kontext mit einzubeziehen)\n")
            user_input = get_user_input().strip().lower()
            prompt = f"Given the following user message, decide what action should be taken. The options are: followup, followup_with_question, newquestion. Choose 'followup' if the user-response indicates a positive sentiment to the Question 'Willst du eine Folgefrage stellen?', If the user response contains already a question you should respond 'followup_with_question'. Choose 'newquestion' if the user does not want to ask a followup question. \n\nUser message: {user_input}\n\nAction:"
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": f"{prompt}"},
                ],
                max_tokens=10,
                stop=['\n']
            )
            print("\nFollowup decision: ")
            action = response.choices[0].message.content
            print(action)
            return action, user_input
        



def get_user_filter():
    print("Welche Mitarbeitergruppe betirfft deine Frage?")
    print("1: Nicht-Ärzliches Personal")
    print("2: Assistenzärzte")
    print("3: Kaderärzte (OA.i.V. mit FA / OA / LA / CA)")
    print("4: Unklar / Keine Angabe\n")
    while True:
        choice = get_user_input()
        if choice == "1":
            filter_values = "ABPR"
            break
        elif choice == "2":
            filter_values = "ARG"
            break
        elif choice == "3":
            filter_values = "KAR"
            break
        else:
            bot_response("Bitte zwischen 1-3 wählen!")
    print("\n Filter chosen: " + str(filter_values) + "\n")
    return filter_values    


def initiate_conversation(user_id):
    role = f"Du bist ein HR-Assistent welcher Fragen von Angestellten des Stadtspitals Zürich beantwortet."
    context = f"Begrüsse den User mit 'Willkommen bei STZ Chat-Bot' und wenn vorhanden mit seiner ID {user_id} und initiere die Konversation, ob der User eine Frage stellen möchte."
    response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"{role}"},
                {"role": "user", "content": f"{context}"},
            ],
            max_tokens=50,
            stop=['\n']
        )
    return response.choices[0].message.content

def add_to_convo(convo, message):
    convo += " "
    convo += message
    return convo



def chat(user_id):
    global keepalive
    keepalive = True
    currentConvo = True
    
    while keepalive:
        convo = ""
        user_input = get_user_input()
        filter = get_user_filter()
        while currentConvo:
            desicion, reply = decide_action(user_input, 1)
            convo = add_to_convo(convo, reply)

            print("\nCurrent convo: " + convo + "\n")

            if desicion == "inquire_more_information":
                response = inquire_more_information(reply)
                convo = add_to_convo(convo, response)

            elif desicion == "perform_rag_request":
                response, articles =  perform_rag_request_with_context(convo, filter, additional_context="")
                #response_2 = perform_rag_request_with_context(convo, filter, additional_context="")
                convo = add_to_convo(convo, response)
                print("\nBOT response with context function: \n")
                bot_response(response)
                #print("\n")
                #print("\nBOT response without context function: \n")
                #bot_response(response_2)
                action, user_input = decide_action(response, 2)   
                if action == "followup":
                    bot_response("Was ist deine Folgefrage?")
                    user_input = get_user_input()
                    convo = add_to_convo(convo, response)
                    continue

                elif action == "followup_with_question":
                    convo = add_to_convo(convo, response)
                    continue

                elif action == "newquestion":
                    break
            else:
                keepalive = False
                break

    c.execute("INSERT INTO logs (user_id, message, response, articles, action) VALUES (?, ?, ?, ?, ?)",
                        (user_id, convo, reply, articles, action))
    conn.commit()                

   
def chatbot(user_id):
    print("Chatbot initialized. Type 'quit' to exit.")
    bot_response(initiate_conversation(user_id))
    while True:
        chat(user_id)


def main():
    print("Welcome! Please choose an option:")
    print("1. Register")
    print("2. Login")
    choice = input("Enter choice: ")

    username = input("Username: ")
    password = getpass("Password: ")

    if choice == '1':
        register(username, password, c, conn)
        user_id = login(username, password)
        if user_id:
            chatbot(user_id)
    elif choice == '2':
        user_id = login(username, password, c)
        if user_id:
            chatbot(user_id)
    else:
        print("Invalid choice.")



if __name__ == "__main__":
    main()
