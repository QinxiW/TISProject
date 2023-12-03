def greet():
    print("Hello! Welcome to the interactive dialogue.")

def ask_name():
    name = input("What is your name? ")
    print(f"Nice to meet you, {name}!")

def main():
    greet()
    ask_name()

    while True:
        user_input = input("Ask me a question or type 'exit' to end: ")

        if user_input.lower() == 'exit':
            print("Exiting...")
            break

        # Customize the dialogue logic based on user input
        if "how are you" in user_input.lower():
            print("I'm doing well, thank you!")
        else:
            print("I'm not sure how to respond to that.")


if __name__ == "__main__":
    main()