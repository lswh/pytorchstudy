import os
import json
import requests
from flask import Flask, request
from moviebot import bot

app = Flask(__name__)
MOVIE_BOT_CHALLENGE = os.environ.get('FB_CHALLENGE')
FB_PAGE_ACCESS_TOKEN = os.environ.get('FB_PAGE_ACCESS_TOKEN')


@app.route('/webhook', methods=['GET', 'POST'])
def webhook():
    # Webhook verification
    if request.method == 'GET':
        mode = request.args.get('hub.mode')
        token = request.args.get('hub.verify_token')
        challenge = request.args.get('hub.challenge')

        if mode and token:
            if mode == 'subscribe' and token == MOVIE_BOT_CHALLENGE:
                return challenge

    # Incoming message
    elif request.method == 'POST':

        event = request.get_json()
        if event['object'] == 'page':
            for entry in event['entry']:
                try:
                    message_event = entry['messaging'][0]
                    message_handler(message_event['sender']['id'],
                                    message_event['message'])
                except IndexError:
                    pass
        return "ok"


def message_handler(sender_id, message):
    text = message['text']
    answer = bot.tell(text)
    uri = "https://graph.facebook.com/v2.6/me/messages?access_token=%s" % FB_PAGE_ACCESS_TOKEN
    data = {
        'messaging_type': 'RESPONSE',
        'recipient': {'id': sender_id},
        'message': {'text': answer}
    }
    requests.post(uri, data=json.dumps(data),
                  headers={'content-type': 'application/json'})


app.run('0.0.0.0', 9898)