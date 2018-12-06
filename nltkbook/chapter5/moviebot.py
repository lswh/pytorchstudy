import random
from chatbot import Chatbot
from datasets import MOVIE_EXPERT_TRAINING_SET

import tmdbsimple as tmdb
tmdb.API_KEY = 'YOUR-TMDB-V3-API-KEY'
tmdb.API_KEY = 'c27f5d633a1cac6ce972aa465971ef52'

bot = Chatbot()
bot.train(MOVIE_EXPERT_TRAINING_SET)

@bot.register_default_handler()
def default_response(_):
    return "Sorry, I can't help you out with that ..."


@bot.register_handler('greetings')
def say_hello(_):
    return random.choice([
        "Hi!",
        "Hi there!",
        "Hello yourself!"
    ])


@bot.register_handler('bye')
def say_bye(_):
    return random.choice([
        "Goodbye to you!",
        "Nice meeting you, bye",
        "Have a good day!",
        "Goodbye"
    ])


@bot.register_handler('thanks')
def say_thanks(_):
    return random.choice([
        "No, thank you!",
        "It's what I do ...",
        "At your service ...",
        "Come back anytime"
    ])


@bot.register_handler('chit-chat')
def make_conversation(_):
    return random.choice([
        "I'm fine, thanks for asking ...",
        "The weather is fine, business is fine, I'm pretty well myself ...",
        "I'm bored",
        "All good, thanks",
    ])


@bot.register_handler('rude')
def be_rude(_):
    return random.choice([
        "That's pretty rude of you. Come back with some manners ...",
        "You're rude, leave me alone.",
        "That's not very nice ...",
        "You're not a nice person, are you?",
    ])


@bot.register_handler('ask-info')
def provide_info(_):
    return random.choice([
        "I know a looot of movies, try me!",
        "I can help you with movies and stuff ...",
        "I am your father!",
        "I come from the future!",
    ])


@bot.register_handler('get-latest')
def get_latest(_):
    try:
        latest_movies = tmdb.Movies().now_playing()['results']
        random.shuffle(latest_movies)
        return "Here are some new movies I think you're going to like: %s" % (
            ", ".join(['"' + m['original_title'] + '"' for m in latest_movies[:5]]))
    except KeyError:
        return "I'm not as up to date as I used to be. Sorry."


@bot.register_handler('what-director')
def what_director(entities):
    if 'movie' not in entities:
        return "Sorry, I didn't quite catch that."

    search = tmdb.Search()
    search.movie(query=entities['movie'])
    try:
        movie = tmdb.Movies(search.results[0]['id'])
    except (KeyError, IndexError):
        return "That's weird ... I never heard of this movie."

    try:
        director = [p for p in movie.credits()['crew'] if p['job'] == 'Director'][0]
        return "The director of %s is %s" % (movie.info()['title'], director['name'])
    except (KeyError, IndexError):
        return "That's weird ... I don't know this one."


@bot.register_handler('what-release-year')
def what_release_year(entities):
    if 'movie' not in entities:
        return "Sorry, what movie is that again?"

    search = tmdb.Search()
    search.movie(query=entities['movie'])

    try:
        movie = search.results[0]
    except (KeyError, IndexError):
        return "Is that a real movie or did you just make it up?"

    try:
        return "The release date of %s is %s" % (movie['title'], movie['release_date'])
    except (KeyError, IndexError):
        return "That's weird ... I don't know the release date."


@bot.register_handler('get-similar-movies')
def get_similar_movies(entities):
    if 'movie' not in entities:
        return "Sorry, I didn't quite catch that."

    search = tmdb.Search()
    search.movie(query=entities['movie'])

    try:
        movie = tmdb.Movies(search.results[0]['id'])
    except (KeyError, IndexError):
        return "That's weird ... I never heard of this movie."

    try:
        recommendations = movie.similar_movies()['results']
        random.shuffle(recommendations)
        return "Here are a few ones I think you're going to enjoy: %s" % (
            ", ".join(['"' + m['original_title'] + '"' for m in recommendations[:5]]))
    except (KeyError, IndexError):
        return "That's weird ... I don't know this one."


@bot.register_handler('actor-movies')
def get_actor_movies(entities):
    try:
        actor = entities['actor']
    except KeyError:
        return "Sorry, what actor is that?"

    search = tmdb.Search()
    search.person(query=actor)

    try:
        actor = tmdb.People(search.results[0]['id'])
    except (IndexError, KeyError):
        return "That's weird ... I never heard of him/her."

    movies = actor.movie_credits()['cast']

    if not movies:
        return "I don't know of any movies featuring him/her."

    random.shuffle(movies)

    return "Ok, here are some movies %s played in: %s" % (
        "he" if actor.info()['gender'] == 2 else "she",
        ", ".join('"%s" in the role of %s' % (m['title'], m['character']) for m in movies[:5])
    )


@bot.register_handler('movie-cast')
def movie_cast(entities):
    if 'movie' not in entities:
        return "Sorry, I didn't quite catch that."

    search = tmdb.Search()
    search.movie(query=entities['movie'])

    try:
        movie = tmdb.Movies(search.results[0]['id'])
    except (IndexError, KeyError):
        return "That's weird ... I never heard of this movie."

    cast = movie.credits()['cast']
    if not cast:
        return "I don't know of any actors playing in this movie ..."

    return "Here are the main actors: %s" % (
        ", ".join('"%s" playing the role of %s' % (m['name'], m['character']) for m in cast[:5]))


@bot.register_handler('actor-in-movie')
def actor_in_movie(entities):
    try:
        actor, movie = entities['actor'], entities['movie']
    except KeyError:
        return "Did who play in what?"

    search = tmdb.Search()
    search.movie(query=movie)

    try:
        movie = tmdb.Movies(search.results[0]['id'])
    except (IndexError, KeyError):
        return "That's weird ... I never heard of this movie."

    search = tmdb.Search()
    search.person(query=actor)

    try:
        actor = tmdb.People(search.results[0]['id'])
    except (IndexError, KeyError):
        return "That's weird ... I never heard of this actor."

    cast = movie.credits()['cast']
    role = [a for a in cast if a['id'] == actor.info()['id']]

    if not role:
        return 'Nope, I believe "%s" didn\'t play in %s' % (actor.info()['name'], movie.info()['original_title'])

    return 'Yep, "%s" played the role of "%s" in "%s"' % (
        actor.info()['name'], role[0]['character'], movie.info()['original_title']
    )