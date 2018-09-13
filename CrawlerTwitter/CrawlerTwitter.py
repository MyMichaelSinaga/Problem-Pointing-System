# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 10:08:35 2018

@author: Seira
"""

import tweepy
import csv

consumer_key = "PE4XXNh1whOqD3c930zAZa5CB"
consumer_secret = "TH2CB6MNMFeJFIl5MqEF7WNThdBPfQBkfEDpvxqf1ms7EbSju4"
access_token = "915475624902377472-EZ7nD3BnNTVtzwzoIIlrbzbUF37MSlj"
access_token_secret = "MhDu0zCjKcC0lKiem3hLyJ86p0m8iwTIk3qEpat7f1lks"
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth,wait_on_rate_limit=True)

csvFile = open('New_Begal.csv','w')
csvWriter = csv.writer(csvFile)

for tweet in tweepy.Cursor(api.search,q=["begal"],count=5000,
                           lang="id",
                           since="2018-07-31").items():
    csvWriter.writerow([tweet.created_at, tweet.text.encode('ascii', 'ignore')])
print("sukses")