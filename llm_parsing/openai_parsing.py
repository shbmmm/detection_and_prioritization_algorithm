import pandas as pd
from textblob import TextBlob
import geopandas as gpd
import folium
import random
import openai

# Replace with your OpenAI API key
openai.api_key = 'sk-proj-cMgiloWEmRouvQIolWokT3BlbkFJX8azkLEllROnyMjbWodi'

# Function to get color based on sentiment
def get_color(sentiment):
    if sentiment > 0.5:
        return 'darkgreen'
    elif sentiment > 0:
        return 'green'
    elif sentiment == 0:
        return 'gray'
    elif sentiment > -0.5:
        return 'orange'
    else:
        return 'red'

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

# Create DataFrame
df = pd.DataFrame({
    'username': usernames,
    'tweet': random.choices(tweets, k=50),
    'latitude': latitudes,
    'longitude': longitudes
})

# Perform sentiment analysis
df['sentiment'] = df['tweet'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Convert to GeoDataFrame
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude))

# Create a map centered around the coordinates
m = folium.Map(location=[-35.2820, 149.1286], zoom_start=5)

# Add points to the map
for idx, row in gdf.iterrows():
    color = get_color(row['sentiment'])
    folium.Marker(
        location=[row['latitude'], row['longitude']],
        popup=f"{row['username']}: {row['tweet']} (Sentiment: {row['sentiment']})",
        icon=folium.Icon(color=color)
    ).add_to(m)

# Save the map to an HTML file
m.save('geotagged_tweets_map_50_realistic_colors.html')

# Prepare data for OpenAI GPT-4
tweets_text = "\n".join(df['tweet'].tolist())
sentiment_summary = df['sentiment'].describe()

# Create a prompt for GPT-4
prompt = f"""
Based on the following tweets about the Australian bushfires and their sentiments, prepare a report for the rescue team.

Tweets:
{tweets_text}

Sentiment Summary:
{sentiment_summary}

Report:
"""

# Generate the report using OpenAI GPT-3.5 Turbo
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ],
    max_tokens=500
)

report = response['choices'][0]['message']['content'].strip()

# Save the report to a text file
with open('/mnt/data/bushfire_report.txt', 'w') as f:
    f.write(report)

print("Report generated and saved to 'bushfire_report.txt'")
