MOVIE_EXPERT_TRAINING_SET = {
    # General Intents
    'greetings': [
        ("Hello", {}),
        ("Hello!", {}),
        ("Hi!", {}),
        ("How are you?", {}),
        ("Hi There!", {}),
        ("Hello there!!!", {}),
        ("Hi!", {}),
        ("Ello!", {}),
        ("Hey!", {}),
        ("Hello mate, how are you?", {}),
        ("Good morning", {}),
        ("Good morning!", {}),
        ("mornin'", {}),
    ],
    'bye': [
        ("Bye!", {}),
        ("Bye Bye!", {}),
        ("ByeBye!", {}),
        ("Bbye!", {}),
        ("See you!", {}),
        ("See you later", {}),
        ("Good bye", {}),
        ("See you soon", {}),
        ("Talk to you later!", {}),
    ],
    'thanks': [
        ("Thanks dude!", {}),
        ("Thank you!", {}),
        ("Thanks", {}),
        ("10x", {}),
        ("10q", {}),
        ("awesome, thanks for your help", {}),
        ("Thank you so much!", {}),
        ("Cheers!", {}),
        ("Cheers, thanks", {}),
        ("many thanks", {}),
    ],
    'rude': [
        ("You're an idiot", {}),
        ("You're such a fucking idiot!", {}),
        ("You're so stupid !!!", {}),
        ("You stupid fuck !!!", {}),
        ("Shut up!", {}),
        ("Shut the fuck up you fuckhead!", {}),
        ("You're such a mother fucker!", {}),
        ("shit!!", {}),
        ("fuck!!", {}),
        ("fuck you!!", {}),
        ("haha, you stupid !", {}),
    ],
    'chit-chat': [
        ("How are you?", {}),
        ("Hey, how are you?", {}),
        ("How's life?", {}),
        ("How you've been?", {}),
        ("How's your day?", {}),
        ("How's life?", {}),
        ("How's business?", {}),
    ],
    'ask-info': [
        ("Tell me more about yourself", {}),
        ("Who are you?", {}),
        ("Can you provide more info?", {}),
        ("Can I get some information?", {}),
        ("Give me more info", {}),
        ("Give me some info please", {}),
        ("I need some info", {}),
        ("I would like to know more please", {}),
        ("Tell me about you", {}),
        ("Let's talk about you", {}),
        ("Who are you?", {}),
    ],

    # Movie specific intents
    'what-director': [
        ("Who directed The Godfather?",
         {'movie': 'The Godfather'}),
        ("Who is the director of The Dark Knight?",
         {'movie': 'The Dark Knight'}),
        ("Who directed Pulp Fiction?",
         {'movie': 'Pulp Fiction'}),
        ("Who is the director of Lord of the rings?",
         {'movie': 'Lord of the rings'}),
        ("The director of Fight Club is?",
         {'movie': 'Fight Club'}),
        ("director of Forrest Gump is?",
         {'movie': 'Forrest Gump'}),
        ("Who Directed The Matrix?",
         {'movie': 'The Matrix'}),
        ("Who is the director of The Silence of the Lambs?",
         {'movie': 'The Silence of the Lambs'}),
    ],
    'what-release-year': [
        ("In what year was Saving Private Ryan released?",
         {'movie': 'Saving Private Ryan'}),
        ("In what year was Interstellar released?",
         {'movie': 'Interstellar'}),
        ("When was Psycho released?",
         {'movie': 'Psycho'}),
        ("When was Casablanca released?",
         {'movie': 'Casablanca'}),
        ("When was Raiders of the Lost Ark produced?",
         {'movie': 'Raiders of the Lost Ark'}),
        ("What's the release year for The Pianist?",
         {'movie': 'The Pianist'}),
        ("What's the release year for The Matrix?",
         {'movie': 'The Matrix'}),
    ],
    'get-similar-movies': [
        ("What are some similar movies to Interstellar",
         {'movie': 'Interstellar'}),
        ("What are some similar movies to Rear Window ?",
         {'movie': 'Rear Window'}),
        ("Can I get some similar movies to Back to the Future ?",
         {'movie': 'Back to the Future'}),
        ("I want more movies like Back to the Future ?",
         {'movie': 'Back to the Future'}),
        ("I want more movies like Django Unchained ?",
         {'movie': 'Django Unchained'}),
        ("I want more movies similar to Blade Runner ?",
         {'movie': 'Blade Runner'}),
        ("I want to know more movies similar to American Beauty ?",
         {'movie': 'American Beauty'}),
        ("Get me more movies like Citizen Kane ?",
         {'movie': 'Citizen Kane'}),
        ("What are some movies similar to Citizen Kane ?",
         {'movie': 'Citizen Kane'}),
    ],
    'get-latest': [
        ("What are some new movies?", {}),
        ("What are some new released movies?", {}),
        ("What are some new releases?", {}),
        ("Can I get some new movies?", {}),
        ("What's new?", {}),
        ("What are some new and cool movies?", {}),
        ("Tell me about some new movies?", {}),
        ("Tell me about new releases?", {}),
    ],
    'actor-movies': [
        ('In what movies did Ryan Gosling acted?',
         {'actor': 'Ryan Gosling'}),
        ('in which movies did Selena Gomez play?',
         {'actor': 'Selena Gomez'}),
        ('In which films did Jason Momoa play?',
         {'actor': 'Jason Momoa'}),
        ('What are some movies in which Emilia Clarke acted ?',
         {'actor': 'Emilia Clarke'}),
        ('What are some movies which featured Johnny Depp ?',
         {'actor': 'Johnny Depp'}),
        ('Tell me some movies with Johnny Depp ?',
         {'actor': 'Johnny Depp'}),
        ('Tell me some movies featuring Tom Cruise ?',
         {'actor': 'Tom Cruise'}),
        ('Get me some movies featuring Leonardo DiCaprio ?',
         {'actor': 'Leonardo DiCaprio'}),
        ('Some films with Matt Damon ?',
         {'actor': 'Matt Damon'}),
        ('Some cool films with Keanu Reeves ?',
         {'actor': 'Keanu Reeves'}),
    ],
    'movie-cast': [
        ("What's the cast of Back to the Future ?",
         {'movie': 'Back to the Future'}),
        ("What's the cast of Forrest Gump?",
         {'movie': 'Forrest Gump'}),
        ("Who acted in The Matrix?",
         {'movie': 'The Matrix'}),
        ("Who acted in American Beauty ?",
         {'movie': 'The Matrix'}),
        ("Who played in Django Unchained?",
         {'movie': 'Django Unchained'}),
        ("What are the actors playing in Pulp Fiction?",
         {'movie': 'Pulp Fiction'}),
        ("Give me the cast of Pulp Fiction?",
         {'movie': 'Pulp Fiction'}),
        ("Tell me the full cast of The Silence of the Lambs?",
         {'movie': 'The Silence of the Lambs'}),
        ("Who was casted in Citizen Kane ?",
         {'movie': 'Citizen Kane'}),
        ("Who was casted in Rear Window ?",
         {'movie': 'Rear Window'}),
    ],
    'actor-in-movie': [
        ("Did Morgan Freeman play in The Shawshank Redemption movie?",
         {'actor': 'Morgan Freeman', 'movie': 'The Shawshank Redemption'}),
        ("Did Marlon Brando act in The Godfather?",
         {'actor': 'Marlon Brando', 'movie': 'The Godfather'}),
        ("Was Al Pacino an actor in The Godfather?",
         {'actor': 'Al Pacino', 'movie': 'The Godfather'}),
        ("Is Uma Thurman the actress in Pulp Fiction?",
         {'actor': 'Uma Thurman', 'movie': 'Pulp Fiction'}),
        ("Was Brad Pitt the actor in Fight Club?",
         {'actor': 'Brad Pitt', 'movie': 'Fight Club'}),
        ("Was Ingrid Bergman in the movie Casablanca?",
         {'actor': 'Ingrid Bergman', 'movie': 'Casablanca'}),
        ("Did Russell Crowe star in Gladiator?",
         {'actor': 'Russell Crowe', 'movie': 'Gladiator'}),
        ("Was Sigourney Weaver a star in the Alien movie?",
         {'actor': 'Sigourney Weaver', 'movie': 'Alien'}),
        ("Did Kevin Spacey act in American Beauty?",
         {'actor': 'Kevin Spacey', 'movie': 'American Beauty'}),
    ]
}