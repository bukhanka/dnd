import streamlit as st
import os
from typing import Dict, Tuple
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from openai import OpenAI
import time
import json
import random
import requests
import base64

load_dotenv()

st.set_page_config(page_title="TD-LLM-DND", page_icon="🐉", layout="wide")

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PDF_FOLDER = os.getenv('PDF_FOLDER', 'pdf')
CHROMA_DB_DIR = os.getenv('CHROMA_DB_DIR', './chroma_db')
TURN_LIMIT = int(os.getenv('TURN_LIMIT', 10))
MODEL_NAME = "gpt-4o-2024-08-06"

for dir in [PDF_FOLDER, CHROMA_DB_DIR]:
    os.makedirs(dir, exist_ok=True)

client = OpenAI(api_key=OPENAI_API_KEY)

def set_fantasy_theme():
    st.markdown("""
    <style>
        body { color: #e0e0e0; background-color: #1a1a2e; font-family: 'Cinzel', serif; }
        .stButton>button { color: #ffd700; background-color: #4a0e0e; border: 2px solid #ffd700; }
        .stTextInput>div>div>input, .stTextArea>div>div>textarea { color: #e0e0e0; background-color: #2a2a4e; }
        .stHeader { color: #ffd700; text-shadow: 2px 2px 4px #000000; }
        .sidebar .sidebar-content { background-color: #16213e; }
        .stApp > header {
            background-color: #4a0e0e;
            background-image: url('https://example.com/fantasy-banner.jpg');
            background-size: cover;
        }
        .stApp > header .decoration {
            display: none;
        }
    </style>
    <link href="https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700&display=swap" rel="stylesheet">
    """, unsafe_allow_html=True)


@st.cache_resource
def initialize_rag():
    embedding_model = OpenAIEmbeddings()
    return Chroma(embedding_function=embedding_model, persist_directory=CHROMA_DB_DIR)


def check_openai_availability():
    try:
        response = client.models.retrieve(MODEL_NAME)
        return True
    except Exception:
        return False


def api_call(prompt: str, max_tokens: int, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens
            )
            if not response.choices:
                raise ValueError("No response choices returned")
            return response.choices[0].message.content
        except Exception as e:
            if attempt == retries - 1:
                st.error(f"API call error after {retries} attempts: {str(e)}")
                return f"Error: Unable to generate content. Please check OpenAI API status."
            time.sleep(2 ** attempt)  # Exponential backoff


@st.cache_data(ttl=3600)
def generate_character() -> str:
    return api_call("Generate a D&D character with name, race, class, backstory, and items. Be creative and diverse.", 150)


def customize_character(name: str, info: str) -> str:
    st.subheader(f"Customize {name}")
    race = st.selectbox("Race", ["Human", "Elf", "Dwarf", "Halfling", "Orc"], key=f"{name}_race")
    char_class = st.selectbox("Class", ["Warrior", "Mage", "Rogue", "Cleric", "Ranger"], key=f"{name}_class")
    backstory = st.text_area("Backstory", key=f"{name}_backstory")
    return f"Name: {name}\nRace: {race}\nClass: {char_class}\nBackstory: {backstory}"


def generate_party() -> Dict[str, str]:
    party = {f"Player {i + 1}": generate_character() for i in range(4)}
    for name, info in party.items():
        party[name] = customize_character(name, info)
    return party


def start_new_adventure(party_members: Dict[str, str], difficulty: str) -> Tuple[Dict, str]:
    dm_intro = api_call(f"You are the Dungeon Master. Start an exciting and unique D&D adventure with {difficulty} difficulty. Introduce the characters: {', '.join(party_members.keys())}. Set the scene and present an initial challenge or mystery.", 300)
    return {
        "turn": 1,
        "difficulty": difficulty,
        "story_progression": [dm_intro],
        "turn_participation": {name: False for name in party_members},
        "party_members": party_members
    }, dm_intro


def load_images():
    image_dir = "images"
    images = {}
    for filename in os.listdir(image_dir):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            name = os.path.splitext(filename)[0]
            images[name] = st.image(os.path.join(image_dir, filename))
    return images


def generate_music_with_suno(prompt: str) -> str:
    base_url = os.getenv('SUNO_API_ENDPOINT')  # Move to environment variable
    url = f"{base_url}/api/generate"
    payload = {
        "prompt": prompt,
        "make_instrumental": False,
        "wait_audio": True
    }
    try:
        response = requests.post(url, json=payload, headers={'Content-Type': 'application/json'}, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data[0]['audio_url']
    except requests.RequestException as e:
        st.error(f"Error generating music: {str(e)}")
        return None

def play_sound(sound_name: str, url: str = None):
    if url:
        st.audio(url, format="audio/mp3")
    else:
        audio_file = open(f"sounds/{sound_name}.mp3", "rb")
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format="audio/mp3")

def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio id="background-music" autoplay loop>
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(md, unsafe_allow_html=True)

def display_status_sidebar():
    st.sidebar.header("System Status")
    openai_status = "Running" if check_openai_availability() else "Not Running"
    st.sidebar.write(f"OpenAI API Status: {openai_status}")
    st.sidebar.write(f"Model: {MODEL_NAME}")
    st.sidebar.write(f"RAG Initialized: {'Yes' if 'vector_store' in st.session_state else 'No'}")

    st.sidebar.header("Game Controls")
    if st.sidebar.button("🧙‍♂️ New Party"):
        with st.spinner("Summoning brave adventurers..."):
            st.session_state.party = generate_party()
        st.success("Your party has assembled!")
        play_sound("party_assembled")

    difficulty = st.sidebar.select_slider("Difficulty", options=["Easy", "Medium", "Hard"], value="Medium")

    if st.sidebar.button("🗺️ New Adventure"):
        if st.session_state.party is None:
            st.error("Please generate a party before starting an adventure.")
        else:
            with st.spinner("Preparing an epic quest..."):
                st.session_state.game_state, _ = start_new_adventure(st.session_state.party, difficulty)
                
                # Generate music using Suno API
                music_prompt = "Create an epic fantasy adventure theme"
                music_url = generate_music_with_suno(music_prompt)
                
                if music_url:
                    play_sound("adventure_start", url=music_url)
                else:
                    play_sound("adventure_start")
                
                # Reset audio playing state
                st.session_state.audio_playing = False
                st.rerun()  # Changed from st.experimental_rerun()
                
            st.success("Your adventure begins!")

    if st.sidebar.button("🔄 Reset"):
        st.session_state.game_state = None
        st.session_state.party = None
        st.success("Game reset. Ready for a new adventure!")

    if st.sidebar.button("💾 Save Game"):
        save_game_state()
        st.success("Game state saved!")

    if st.sidebar.button("📂 Load Game"):
        loaded_state = load_game_state()
        if loaded_state:
            st.session_state.game_state = loaded_state
            st.success("Game state loaded!")
        else:
            st.warning("No saved game state found.")

def player_turn(player_name: str, player_info: str, game_state: Dict, vector_store) -> str:
    context = vector_store.similarity_search(player_info, k=3)
    player_prompt = f"{player_info}\nGame context: {' '.join(game_state['story_progression'][-3:])}\nRelevant lore: {' '.join([doc.page_content for doc in context])}\nWhat do you do next? (Respond in character)"
    return api_call(player_prompt, 150)


def dm_turn(game_state: Dict, vector_store) -> str:
    context = vector_store.similarity_search(" ".join(game_state['story_progression'][-5:]), k=3)
    dm_prompt = f"As the Dungeon Master, consider the recent events:\n{' '.join(game_state['story_progression'][-5:])}\nRelevant lore: {' '.join([doc.page_content for doc in context])}\nSummarize the actions, introduce the next challenge or plot development, and describe the scene. Be creative and engaging."
    return api_call(dm_prompt, 300)


def add_item_to_inventory(player: str, item: str):
    if 'inventory' not in st.session_state.game_state:
        st.session_state.game_state['inventory'] = {}
    if player not in st.session_state.game_state['inventory']:
        st.session_state.game_state['inventory'][player] = []
    st.session_state.game_state['inventory'][player].append(item)

def main():
    set_fantasy_theme()
    st.title("🐉 TD-LLM-DND")

    # Initialize session state variables
    if 'game_state' not in st.session_state:
        st.session_state.game_state = None
    if 'party' not in st.session_state:
        st.session_state.party = None
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = initialize_rag()
    if 'audio_playing' not in st.session_state:
        st.session_state.audio_playing = False

    # Update this line to use a file from your sounds directory
    if not st.session_state.audio_playing:
        autoplay_audio("sounds/adventure_start.mp3")
        st.session_state.audio_playing = True

    tab1, tab2, tab3 = st.tabs(["Game", "Party", "Inventory"])

    with tab1:
        display_game_controls()
        display_adventure_log()

    with tab2:
        display_party_information()

    with tab3:
        display_inventory()

    if not check_openai_availability():
        st.error(f"OpenAI API or the model '{MODEL_NAME}' is not available. Please make sure you have a valid API key and the model is accessible.")
        st.info("To get an OpenAI API key, sign up for an account on the OpenAI website.")
        if st.button("Retry Connection"):
            st.experimental_rerun()
        return

    if st.session_state.game_state is None and st.session_state.party is None:
        st.info("Generate a party and start a new adventure to begin your journey!")

    st.sidebar.info("""
    ## How to Play
    1. Generate New Party
    2. Start New Adventure
    3. Play Next Turn

    May your dice roll high!
    """)

    st.sidebar.header("🎒 Inventory")
    if st.session_state.game_state and 'inventory' in st.session_state.game_state:
        for player, items in st.session_state.game_state['inventory'].items():
            with st.sidebar.expander(f"{player}'s Items"):
                for item in items:
                    st.sidebar.markdown(f"- {item}")
    else:
        st.sidebar.write("No items in inventory yet.")

    if st.session_state.game_state:
        st.sidebar.markdown(f"### Turn: {st.session_state.game_state['turn']}/{TURN_LIMIT}")
        progress = st.session_state.game_state['turn'] / TURN_LIMIT
        st.sidebar.progress(progress)

    # In your sidebar
    if st.sidebar.checkbox("Play Background Music", value=True):
        autoplay_audio("sounds/adventure_start.mp3")

def display_party_information():
    if st.session_state.party:
        st.header("🦸‍♀️ Party Information")
        for name, info in st.session_state.party.items():
            with st.expander(f"⚔️ {name}"):
                st.write(info)

def display_adventure_log():
    if st.session_state.game_state is not None:
        st.header("📜 Adventure Log")
        for i, event in enumerate(st.session_state.game_state["story_progression"]):
            with st.container():
                st.markdown(f"<div style='background-color: #2a2a4e; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>{event}</div>", unsafe_allow_html=True)

        if "current_scene" in st.session_state.game_state:
            st.image(st.session_state.images.get(st.session_state.game_state["current_scene"], "default_scene.jpg"))

        if st.button("🎲 Play Next Turn"):
            play_sound("dice_roll")
            progress_bar = st.progress(0)
            with st.spinner("The dice are rolling..."):
                total_steps = len(st.session_state.game_state["party_members"]) + 1
                current_step = 0
                for player_name, player_info in st.session_state.game_state["party_members"].items():
                    if not st.session_state.game_state["turn_participation"][player_name]:
                        action = player_turn(player_name, player_info, st.session_state.game_state, st.session_state.vector_store)
                        st.session_state.game_state["story_progression"].append(f"{player_name}: {action}")
                        st.session_state.game_state["turn_participation"][player_name] = True
                        st.text_area(player_name, action, height=100, disabled=True)

                    current_step += 1
                    progress_bar.progress(current_step / total_steps)

                dm_response = dm_turn(st.session_state.game_state, st.session_state.vector_store)
                st.session_state.game_state["story_progression"].append(dm_response)
                st.text_area("Dungeon Master", dm_response, height=150, disabled=True)

                current_step += 1
                progress_bar.progress(1.0)

                st.session_state.game_state["turn"] += 1
                st.session_state.game_state["turn_participation"] = {name: False for name in st.session_state.game_state["turn_participation"]}

                if st.session_state.game_state["turn"] > TURN_LIMIT:
                    st.warning(f"The adventure has reached its end after {TURN_LIMIT} turns. Start a new game to continue playing!")

            progress_bar.empty()
            play_sound("turn_complete")
    else:
        st.info("No active game. Start a new adventure to begin!")

def display_inventory():
    if st.session_state.game_state and 'inventory' in st.session_state.game_state:
        for player, items in st.session_state.game_state['inventory'].items():
            with st.expander(f"{player}'s Items"):
                for item in items:
                    st.markdown(f"- {item}")
    else:
        st.write("No items in inventory yet.")

def display_game_controls():
    st.header("Game Controls")
    if st.button("🧙‍♂️ New Party"):
        with st.spinner("Summoning brave adventurers..."):
            st.session_state.party = generate_party()
        st.success("Your party has assembled!")
        play_sound("party_assembled")

    difficulty = st.select_slider("Difficulty", options=["Easy", "Medium", "Hard"], value="Medium")

    if st.button("🗺️ New Adventure"):
        if st.session_state.party is None:
            st.error("Please generate a party before starting an adventure.")
        else:
            with st.spinner("Preparing an epic quest..."):
                st.session_state.game_state, _ = start_new_adventure(st.session_state.party, difficulty)
                
                # Generate music using Suno API
                music_prompt = "Create an epic fantasy adventure theme"
                music_url = generate_music_with_suno(music_prompt)
                
                if music_url:
                    play_sound("adventure_start", url=music_url)
                else:
                    play_sound("adventure_start")
                
                # Reset audio playing state
                st.session_state.audio_playing = False
                st.rerun()  # Changed from st.experimental_rerun()
                
            st.success("Your adventure begins!")

    if st.button("🔄 Reset"):
        st.session_state.game_state = None
        st.session_state.party = None
        st.success("Game reset. Ready for a new adventure!")

    if st.button("💾 Save Game"):
        save_game_state()
        st.success("Game state saved!")

    if st.button("📂 Load Game"):
        loaded_state = load_game_state()
        if loaded_state:
            st.session_state.game_state = loaded_state
            st.success("Game state loaded!")
        else:
            st.warning("No saved game state found.")

def save_game_state():
    if st.session_state.game_state:
        with open('game_state.json', 'w') as f:
            json.dump(st.session_state.game_state, f)

def load_game_state():
    if os.path.exists('game_state.json'):
        with open('game_state.json', 'r') as f:
            return json.load(f)
    return None

if __name__ == "__main__":
    main()