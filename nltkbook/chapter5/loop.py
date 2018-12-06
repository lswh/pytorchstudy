from moviebot import bot


while True:
    line = input("You: ")
    print("MovieBot: ", bot.tell(line))