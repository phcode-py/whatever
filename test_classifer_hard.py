from reddit_opinn import build_sentiment_classifier, score_sentiment
classifier = build_sentiment_classifier()
TEXT= "i am against abortion but i am also against banning it. i think it should be legal but not encouraged. i think it should be a last resort and not a first option. i think it should be safe and accessible for those who need it, but also that we should do more"

score_sentiment(classifier,TEXT)
print(f"Sentiment score for: '{TEXT}' is {score_sentiment(classifier,TEXT)}")
