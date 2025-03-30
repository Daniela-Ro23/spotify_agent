import json
import openai
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import os
from dotenv import load_dotenv

from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI


load_dotenv()

# Authenticate Spotify API
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=os.getenv('SPOTIPY_CLIENT_ID'),
                                               client_secret=os.getenv('CLIENT_SECRET'),
                                               redirect_uri='http://localhost/',
                                               scope='playlist-modify-public playlist-modify-private playlist-read-private'))

# Define a TypedDict named 'State' to represent a structured dictionary
class State(TypedDict):
    user_input: str
    intent: Optional[str]
    playlist_name: Optional[str]
    song_name: Optional[str] 
    artist_name: Optional[str] 
    num_songs: int
    mood: Optional[str]
    tracks: List[str]
    playlist_uri: str

# Define LLM Agent
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
client = openai.OpenAI()


# Define OpenAI function schema
query_extraction_function = {
    "name": "extract_query_details",
    "description": "Extract structured details from a user's playlist request.",
    "parameters": {
        "type": "object",
        "properties": {
            "intent": {"type": "string", "enum": ["find_songs", "create_playlist"], "description": "User's intent"},
            "playlist_name": {"type": "string", "description": "Playlist name"},
            "song_name": {"type": "string", "description": "Song name"},
            "artist_name": {"type": "string", "description": "Artist name"},
            "num_songs": {"type": "integer", "description": "Number of songs"},
            "mood": {"type": "string", "description": "Mood or energy level"}
        },
        "required": ["intent"]
    }
}


# Define workflow nodes
def extract_user_query(state: State) -> State:
    """Use OpenAI function calling to extract structured details from user input."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": state["user_input"]}],
        functions=[query_extraction_function],
        function_call="auto"
    )

    # Parse function call response
    function_call = response.choices[0].message.function_call
    if function_call:
        function_args = json.loads(function_call.arguments)
        print('function call successful')
        print(function_args)
        return {**state, **function_args}
    
    print('no function call')
    return state  # Return unchanged if no function call occurred


def get_similar_songs(state: State) -> State:
    """Find similar songs based on a track name and artist."""
    song_name = state.get("song_name")
    artist_name = state.get("artist_name")
    if not song_name or not artist_name:
        return state  # No song name provided
    
    num_songs = state.get("num_songs")
    if num_songs:
        limit = state["num_songs"]
    else:
        limit = 10
    
    results = sp.search(q=f"track:{state['song_name']} artist:{state['artist_name']}", type="track", limit=1)

    if not results["tracks"]["items"]:
        return state
    
    track_id = results["tracks"]["items"][0]["id"]
    print(track_id)
    print(results["tracks"]["items"][0]["name"])
    recommendations = sp.recommendations(seed_tracks=[track_id], limit=limit)

    song_ids = [track["id"] for track in recommendations["tracks"]]

    return {"tracks": song_ids}


def filter_songs_by_energy(state: State) -> State:
    """Filter recommended songs based on energy level."""
    song_name = state.get("song_name")
    mood = state.get("mood")
    if not song_name or not mood:
        return state

    features = sp.audio_features(state["song_ids"])
    filtered_songs = [f["id"] for f in features if f and f["energy"] >= 0.7]

    return {"song_ids": filtered_songs}


def create_playlist(state: State) -> State:
    """Create a playlist and add the recommended songs."""
    user_id = sp.current_user()["id"]
    playlist = sp.user_playlist_create(user_id, state["playlist_name"], public=False)
    
    song_ids = state.get("song_ids")
    if song_ids:
        sp.playlist_add_items(playlist["id"], state["song_ids"])
    
    return {"playlist_url": playlist["external_urls"]["spotify"]}

# Create StateGraph
graph = StateGraph(State)

# Add nodes
graph.add_node("extract_intent", extract_user_query)
graph.add_node("get_similar_songs", get_similar_songs)
graph.add_node("filter_songs", filter_songs_by_energy)
graph.add_node("create_playlist", create_playlist)

# Define edges
graph.set_entry_point("extract_intent")
graph.add_edge("extract_intent", "get_similar_songs")
graph.add_edge("get_similar_songs", "filter_songs")
graph.add_edge("filter_songs", "create_playlist")

graph = graph.compile()

user_query = "Create a playlist called This works."

state = {"user_input": user_query}

final_state = graph.invoke(state)