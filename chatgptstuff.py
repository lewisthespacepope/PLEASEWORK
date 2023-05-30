import openai
import smtplib
import pyperclip


API_KEY = open("api key.txt", "r").read()
openai.api_key = API_KEY


def generate_response(prompt):
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.7
    )
    return response.choices[0].text.strip()


def send_email(content):
    sender_email = 'lewisreynolds18@outlook.com'
    receiver_email = 'lewisthespacepope2001@outlook.com'
    subject = 'Generated Response'
    message = 'Subject: {}\n\n{}'.format(subject, content)

    with smtplib.SMTP('smtp.example.com', 587) as server:
        server.starttls()
        server.login(sender_email, '!L0v3r1pt1d32')
        server.sendmail(sender_email, receiver_email, message)




prompt = 'Your video-related question or statement'

response = generate_response(prompt)
send_email(response)
pyperclip.copy(response)