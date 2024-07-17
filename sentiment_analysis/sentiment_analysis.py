import pandas as pd
from textblob import TextBlob
import geopandas as gpd
import folium
import random

# Create synthetic geotagged Twitter data for Australian bushfire regions with 50 tweets
usernames = [f'user{i}' for i in range(1, 51)]
tweets = [
    'The bushfire is devastating!', 'I hope everyone is safe from the fire.', 'The fire has spread to new areas.',
    'The response team is doing a great job!', 'Praying for the affected families.', 'The smoke is unbearable.',
    'Firefighters are heroes!', 'The fire is out of control!', 'Stay safe everyone.', 'Terrible disaster!',
    'This is so sad.', 'Wishing for everyone\'s safety.', 'Heartbreaking scenes from the bushfire.',
    'The fire is massive.', 'The damage is enormous.', 'Hoping for a quick end to the fire.',
    'The fire fighters are doing an amazing job.', 'The sky is covered in smoke.', 'Stay strong everyone.',
    'The fire keeps spreading.', 'The situation is dire.', 'So much destruction.', 'Unbelievable damage.',
    'The fire is relentless.', 'Can’t believe how bad it is.', 'Sending prayers.', 'The bushfire is terrifying.',
    'This is a tragedy.', 'Hoping for rain to stop the fire.', 'The fire is uncontrollable.',
    'Everyone stay safe.', 'The fire is getting worse.', 'It’s heartbreaking to see this.',
    'The fire is a disaster.', 'Hope the fire ends soon.', 'Feeling so sad about the fire.',
    'It’s awful to see the fire spreading.', 'This fire is a nightmare.', 'The devastation is huge.',
    'Wishing for everyone’s safety.', 'This fire needs to stop.', 'The fire has destroyed so much.',
    'Can’t imagine the pain of those affected.', 'Hoping for a miracle.', 'The fire is beyond control.',
    'Feeling hopeless about the fire.', 'This is an emergency.', 'The fire is catastrophic.',
    'Sending strength to everyone.', 'The fire is shocking.'
]

# Define latitude and longitude points within the affected regions
latitude_range = (-38.5, -32.0)  # Approximate latitude range for affected regions
longitude_range = (140.0, 151.0)  # Approximate longitude range for affected regions

latitudes = [random.uniform(latitude_range[0], latitude_range[1]) for _ in range(50)]
longitudes = [random.uniform(longitude_range[0], longitude_range[1]) for _ in range(50)]


df = pd.DataFrame({
    'username': usernames,
    'tweet': random.choices(tweets, k=50),
    'latitude': latitudes,
    'longitude': longitudes
})

# Perform sentiment analysis
df['sentiment'] = df['tweet'].apply(lambda x: TextBlob(x).sentiment.polarity)


def get_color(sentiment):
    if sentiment <= -0.6:
        return 'darkred'
    elif -0.6 < sentiment <= -0.2:
        return 'red'
    elif -0.2 < sentiment < 0:
        return 'orange'
    elif sentiment == 0:
        return 'lightgray'
    elif 0 < sentiment <= 0.2:
        return 'lightgreen'
    elif 0.2 < sentiment <= 0.6:
        return 'green'
    else:
        return 'darkgreen'


gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude))

# Create a map centered around the coordinates
m = folium.Map(location=[-35.2820, 149.1286], zoom_start=5)


for idx, row in gdf.iterrows():
    color = get_color(row['sentiment'])
    folium.Marker(
        location=[row['latitude'], row['longitude']],
        popup=f"{row['username']}: {row['tweet']} (Sentiment: {row['sentiment']})",
        icon=folium.Icon(color=color)
    ).add_to(m)

# Save the map to an HTML file
m.save('tweet_distribution.html')

