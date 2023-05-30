import openai

API_KEY = open("api key.txt", "r").read()
openai.api_key = API_KEY

response = openai.ChatCompletion.create(
    model = 'gpt-3.5-turbo',
    messages = [{"role": "user", "content": "what is the difference between bread and carrots?"}]
)


print(response)




