import random
import requests


def search_wikipedia(query):
    base_url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": query,
    }

    response = requests.get(base_url, params=params)
    data = response.json()

    if "query" in data and "search" in data["query"]:
        search_results = data["query"]["search"]
        if search_results:
            return search_results[0]["snippet"]

    return "Sorry, I couldn't find any information on that."


def get_response(user_input):
    if user_input.startswith("search"):
        query = user_input[7:]
        return search_wikipedia(query)
    else:
        responses = {
            "hello": "Hi there!",
            "how are you": "I'm just a computer program, but thanks for asking!",
            "bye": "Goodbye! Have a great day!",
            # Add more responses as needed
        }

        return responses.get(user_input, "I'm not sure how to respond to that.")


def main():
    print("Welcome to the Python Chatbot!")

    while True:
        user_input = input("You: ").lower()

        if user_input == 'exit':
            print("Goodbye! Exiting the chatbot.")
            break

        response = get_response(user_input)
        print("Bot:", response)


if __name__ == "__main__":
    main()
