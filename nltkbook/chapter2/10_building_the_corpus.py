# If you have specific needs for your corpus, remember to adjust these categories and keywords accordingly.
CATEGORIES = {
    'business': [
        "Business", "Marketing", "Management"
    ],
    'family': [
        "Family", "Children", "Parenting"
    ],
    'politics': [
        "Politics", "Presidential Elections",
        "Politicians", "Government", "Congress"
    ],
    'sport': [
        "Baseball", "Basketball", "Running", "Sport",
        "Skiing", "Gymnastics", "Tenis", "Football", "Soccer"
    ],
    'health': [
        "Health", "Weightloss", "Wellness", "Well being",
        "Vitamins", "Healthy Food", "Healthy Diet"
    ],
    'economics': [
        "Economics", "Finance", "Accounting"
    ],
    'celebrities': [
        "Celebrities", "Showbiz"
    ],
    'medical': [
        "Medicine", "Doctors", "Health System",
        "Surgery", "Genetics", "Hospital"
    ],
    'science & technology': [
        "Galaxy", "Physics",
        "Technology", "Science"
    ],
    'information technology': [
        "Artificial Intelligence", "Search Engine",
        "Software", "Hardware", "Big Data",
        "Analytics", "Programming"
    ],
    'education': [
        "Education", "Students", "University"
    ],
    'media': [
        "Newspaper", "Reporters", "Social Media"
    ],
    'cooking': [
        "Cooking", "Gastronomy", "Cooking Recipes",
        "Paleo Cooking", "Vegan Recipes"
    ],
    'religion': [
        "Religion", "Church", "Spirituality"
    ],
    'legal': [
        "Legal", "Lawyer", "Constitution"
    ],
    'history': [
        "Archeology", "History", "Middle Ages"
    ],
    'nature & ecology': [
        "Nature", "Ecology",
        "Endangered Species", "Permaculture"
    ],
    'travel': [
        "Travel", "Tourism", "Globetrotter"
    ],
    'meteorology': [
        "Tornado", "Meteorology", "Weather Prediction"
    ],
    'automobiles': [
        "Automobiles", "Motorcycles", "Formula 1", "Driving"
    ],
    'art & traditions': [
        "Art", "Artwork", "Traditions",
        "Artisan", "Pottery", "Painting", "Artist"
    ],
    'beauty & fashion': [
        "Beauty", "Fashion", "Cosmetics", "Makeup"
    ],
    'relationships': [
        "Relationships", "Relationship Advice",
        "Marriage", "Wedding"
    ],
    'astrology': [
        "Astrology", "Zodiac", "Zodiac Signs", "Horoscope"
    ],
    'diy': [
        'Gardening', 'Construction', 'Decorating',
        'Do it Yourself', 'Furniture'
    ]
}

import uuid
import atexit
import urllib
import random
import requests
import pandas as pd
from time import sleep, time
from bs4 import BeautifulSoup
from newspaper import Article, ArticleException


POCKET_BASE_URL = 'https://getpocket.com/explore/%s'

df = pd.DataFrame(columns=['title', 'excerpt', 'url', 'file_name', "keyword", "category"])

@atexit.register
def save_dataframe():
    """ Before exiting, make sure we save the dataframe to a CSV file """
    dataframe_name = "dataframe_{0}.csv".format(time())
    df.to_csv(dataframe_name, index=False)

# Shuffle the categories to make sure we are not exhaustively crawling only the first categories
categories = list(CATEGORIES.items())
random.shuffle(categories)

for category_name, keywords in categories:
    print("Exploring Category=\"{0}\"".format(category_name))
    for kw in keywords:
        # Get trending content from Pocket's explore endpoint
        result = requests.get(POCKET_BASE_URL % urllib.parse.quote_plus(kw))

        # Extract the media items
        soup = BeautifulSoup(result.content, "html5lib")
        media_items = soup.find_all(attrs={'class': 'media_item'})
        for item_html in media_items:
            title_html = item_html.find_all(attrs={'class': 'title'})[0]
            title = title_html.text


            url = title_html.a['data-saveurl']
            print("Indexing article: \"{0}\" from \"{1}\"".format(title, url))

            excerpt = item_html.find_all(attrs={'class': 'excerpt'})[0].text

            try:
                article = Article(url)
                article.download()
                article.parse()
                content = article.text
            except ArticleException as e:
                print("Encoutered exception when parsing \"{0}\": \"{1}\"".format(url, str(e)))
                continue

            if not content:
                print("Couldn't extract content from \"{0}\"".format(url))
                continue

            # Save the text file
            file_name = "{0}.txt".format(str(uuid.uuid4()))
            with open('./data/files/{0}'.format(file_name), 'w+') as text_file:
                text_file.write(content)

            # Append the row in our dataframe
            df.loc[len(df)] = [title, excerpt, url, file_name, kw, category_name]
            # Need to sleep in order to not get blocked
            sleep(random.randint(5, 15))
