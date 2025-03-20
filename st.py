import streamlit as st
import pandas as pd
import pickle

# Load Pickle Files
@st.cache_data
def load_model():
    with open("kmeans_model.pkl", "rb") as model_file:
        kmeans = pickle.load(model_file)
    with open("scaler.pkl", "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
    return kmeans, scaler

# Load Processed Dataset
@st.cache_data
def load_data():
    df = pd.read_csv("spotify_clustered.csv")

    # Ensure correct column names
    if "artist(s)_name" in df.columns:
        df.rename(columns={"artist(s)_name": "artist_name"}, inplace=True)

    # Ensure 'streams' is numeric
    df['streams'] = pd.to_numeric(df['streams'], errors='coerce')
    df['streams'] = df['streams'].fillna(0).astype(int)

    return df

# Load Model, Scaler, and Data
kmeans, scaler = load_model()
df = load_data()

# 🎨 Custom CSS for Spotify-Like UI
st.markdown("""
    <style>
        body { background-color: #121212; color: white; font-family: 'Arial', sans-serif; }
        .song-card {
            background: linear-gradient(135deg, #1DB954, #1ed760);
            border-radius: 15px;
            padding: 20px;
            margin: 10px 0;
            color: white;
            font-family: Arial, sans-serif;
            box-shadow: 4px 4px 20px rgba(0, 0, 0, 0.3);
            transition: transform 0.3s ease-in-out;
        }
        .song-card:hover { transform: scale(1.05); }
        .song-title { font-size: 22px; font-weight: bold; margin-bottom: 5px; }
        .artist-name { font-size: 18px; font-style: italic; opacity: 0.9; }
        .details { font-size: 15px; opacity: 0.85; margin-top: 5px; }
    </style>
""", unsafe_allow_html=True)

def recommend_music(user_input, df, num_recommendations=7):
    user_input = user_input.lower().strip()

    # Check if input is a track name
    song_match = df[df['track_name'].str.lower() == user_input]
    if not song_match.empty:
        song_name = song_match.iloc[0]['track_name']
        st.success(f"🎵 Showing results for: **{song_name}**")

        # Get similar songs using clustering
        similar_songs = df[df['Cluster'] == song_match['Cluster'].values[0]].sample(num_recommendations - 1)
        final_output = pd.concat([song_match, similar_songs])

        return final_output[['track_name', 'artist_name', 'streams', 'released_year']]

    # Check if input is an artist name
    artist_match = df[df['artist_name'].str.lower() == user_input]
    if not artist_match.empty:
        artist_name = artist_match.iloc[0]['artist_name']
        st.success(f"🎤 Showing songs by artist: **{artist_name}**")

        num_songs = min(len(artist_match), num_recommendations)
        return artist_match[['track_name', 'artist_name', 'streams', 'released_year']].sample(num_songs, replace=False)

    # If no match is found, recommend **random trending songs**
    st.warning("⚠️ No exact match found! Here are some trending songs for you:")
    return df[['track_name', 'artist_name', 'streams', 'released_year']].sample(num_recommendations)

# Streamlit UI
st.title("🎧 Spotify Clone - Music Recommendation")

# User Input Search Box
user_input = st.text_input("🔍 Search for a Track or Artist:", "")

# Initialize recommendations as None (to avoid NameError)
recommendations = None  

if st.button("🎵 Recommend", key="recommend-btn"):
    if user_input:
        recommendations = recommend_music(user_input, df)

# Display results after button is clicked
if recommendations is not None:
    for _, row in recommendations.iterrows():
        st.markdown(f"""
            <div class="song-card">
                <div class="song-title">{row['track_name']}</div>
                <div class="artist-name">{row['artist_name']}</div>
                <div class="details">🎶 Streams: {int(row['streams']):,} | 📅 Year: {int(row['released_year'])}</div>
            </div>
        """, unsafe_allow_html=True)

