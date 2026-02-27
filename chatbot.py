import google.generativeai as ai

# API KEY
API_KEY = 'AIzaSyDx3BQPGmiBTsrO00cRXCN8jZbmr1MEjf8'

# Configure the API
ai.configure(api_key=API_KEY)

# Start a new model
model = ai.Generativemodel("gemini-pro")
chat = model.start_chat()

# start a conversation
while True:
    message = input ('You: ')
    if message.lower() == "Bye":
        print('Chatbot: GoodBye!')
        break
    response = chat.send_message(message)
    print('chatbot:', response.text)