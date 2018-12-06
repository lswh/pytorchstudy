import pandas as pd
import matplotlib.pyplot as plt


# Initialize a dataframe for storing tweets
df = pd.DataFrame(columns=['tweet', 'source', 'sentiment'])

####################################
#
#  NLTK Twitter Samples
#
####################################
from nltk.corpus import twitter_samples

# Add the positive tweets
for tweet in twitter_samples.strings('positive_tweets.json'):
    df.loc[len(df)] = [tweet, 'nltk.corpus.twitter_samples', 'positive']

for tweet in twitter_samples.strings('negative_tweets.json'):
    df.loc[len(df)] = [tweet, 'nltk.corpus.twitter_samples', 'negative']

####################################
#
#  Twitter Airline Reviews
#
####################################
airline_tweets = pd.read_csv('./Tweets.csv')

# Select only the columns of interest
airline_df = airline_tweets[['text', 'airline_sentiment']]

# Rename the columns to fit the header
airline_df = airline_df.rename(columns={'text': 'tweet', 'airline_sentiment': 'sentiment'})

# Add a constant column as the source
airline_df['source'] = 'https://www.kaggle.com/crowdflower/twitter-airline-sentiment'


####################################
#
#  First GOP Debate Twitter Sentiment
#
####################################
debate_tweets = pd.read_csv('./Sentiment.csv')

# Select only the columns of interest
debate_df = debate_tweets[['text', 'sentiment']]

# Rename the columns to fit the header
debate_df = debate_df.rename(columns={'text': 'tweet'})

# Standardize the sentiment column
debate_df['sentiment'] = debate_df['sentiment'].apply(lambda s: s.lower())

# Add a constant column as the source
debate_df['source'] = 'https://www.kaggle.com/crowdflower/first-gop-debate-twitter-sentiment'

# Put everything together recomputing the index
df = pd.concat([df, airline_df, debate_df], ignore_index=True)
print(df)

# Let's see how many positive/neutral/negative samples we've got
df[['tweet', 'sentiment']].groupby(['sentiment']).count().plot(kind='bar')

# Make sure the plot doesn't immediately disappear
plt.show(block=True)

print("Total tweets: ", len(df))

# Save all data to file
df.to_csv('twitter_sentiment_analysis.csv')
