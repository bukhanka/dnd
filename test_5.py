import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI
import time
import base64
from io import BytesIO
from typing import Dict, Tuple, TypedDict, List, Annotated
import requests
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import ToolNode
import operator
import random

load_dotenv()

st.set_page_config(page_title="TD-LLM-DND", page_icon="🐉", layout="wide")

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PDF_FOLDER = os.getenv('PDF_FOLDER', 'pdf')
CHROMA_DB_DIR = os.getenv('CHROMA_DB_DIR', './chroma_db')
TURN_LIMIT = int(os.getenv('TURN_LIMIT', 10))
MODEL_NAME = "gpt-4o-mini"
IMAGE_FOLDER = os.getenv('IMAGE_FOLDER', 'images')

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


def get_random_image_from_folder() -> str:
    """
    Returns a random image filename from the specified folder.
    """
    image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    if image_files:
        return os.path.join(IMAGE_FOLDER, random.choice(image_files))
    return None


def get_image_for_scene(scene_description: str) -> str:
    return get_random_image_from_folder()


def start_new_adventure(party_members: Dict[str, str], difficulty: str) -> Tuple[Dict, str]:
    dm_intro = api_call(f"You are the Dungeon Master. Start an exciting and unique D&D adventure with {difficulty} difficulty. Introduce the characters: {', '.join(party_members.keys())}. Set the scene and present an initial challenge or mystery.", 300)
    
    # Get a random image for the initial scene
    initial_image_path = get_image_for_scene(dm_intro)
    
    return {
        "turn": 1,
        "difficulty": difficulty,
        "story_progression": [dm_intro],
        "turn_participation": {name: False for name in party_members},
        "party_members": party_members,
        "current_image_path": initial_image_path
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
                    autoplay_audio(music_url)
                else:
                    autoplay_audio("sounds/adventure_start.mp3")
                
                # Reset audio playing state
                st.session_state.audio_playing = True
                st.rerun()
                
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

# Define our state
class StoryState(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage], operator.add]
    characters: List[dict]
    current_scene: str
    story_summary: str

def dungeon_master(state: StoryState) -> StoryState:
    prompt = f"Current scene: {state['current_scene']}\nStory summary: {state['story_summary']}\nGenerate an engaging narrative to start a new fantasy adventure and present initial choices to the player."
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "system", "content": "You are a creative and engaging dungeon master starting a new adventure."},
                  {"role": "user", "content": prompt}]
    )
    narrative = response.choices[0].message.content
    return {
        "messages": state["messages"] + [AIMessage(content=narrative)],
        "characters": state["characters"],
        "current_scene": "adventure_start",
        "story_summary": state["story_summary"] + " The adventure begins."
    }

def story_manager(state: StoryState) -> StoryState:
    user_input = state.get("user_input", "")  # Get user_input from the state
    prompt = f"Current scene: {state['current_scene']}\nUser choice: {user_input}\nUpdate the story arc and determine the next scene."
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "system", "content": "You are a creative story manager for an interactive narrative."},
                  {"role": "user", "content": prompt}]
    )
    result = response.choices[0].message.content
    next_scene, summary_update = result.split('\n', 1)
    return {
        "current_scene": next_scene,
        "story_summary": state["story_summary"] + " " + summary_update,
        "messages": state["messages"] + [HumanMessage(content=user_input), AIMessage(content=f"The story moves to: {next_scene}")]
    }

def character_creation(state: StoryState) -> StoryState:
    prompt = f"Current scene: {state['current_scene']}\nStory summary: {state['story_summary']}\nCreate a new character that fits the current narrative."
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "system", "content": "You are a creative character designer for an interactive story."},
                  {"role": "user", "content": prompt}]
    )
    new_character = response.choices[0].message.content
    return {"characters": state["characters"] + [{"description": new_character}]}

def should_end_story(state: StoryState) -> bool:
    prompt = f"Story summary: {state['story_summary']}\nCurrent scene: {state['current_scene']}\nDetermine if the story should end based on narrative progression and engagement."
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "system", "content": "You are an expert storyteller determining if a story should conclude."},
                  {"role": "user", "content": prompt}]
    )
    should_end = response.choices[0].message.content.lower().startswith('yes')
    return should_end

def initialize_story_graph():
    story_graph = StateGraph(StoryState)
    story_graph.add_node("dungeon_master", dungeon_master)
    story_graph.add_node("story_manager", story_manager)
    story_graph.add_node("character_creation", character_creation)
    story_graph.add_edge("dungeon_master", "story_manager")
    story_graph.add_edge("story_manager", "dungeon_master")
    story_graph.add_edge("story_manager", "character_creation")
    story_graph.add_edge("character_creation", "dungeon_master")
    story_graph.set_entry_point("dungeon_master")
    story_graph.add_conditional_edges(
        "story_manager",
        should_end_story,
        {True: END, False: "dungeon_master"}
    )
    return story_graph.compile()

def main():
    set_fantasy_theme()
    st.title("🐉 TD-LLM-DND")

    # Initialize session state variables
    initialize_session_state()

    # Initialize the input field value
    if "main_user_input" not in st.session_state:
        st.session_state.main_user_input = ""

    # Automatically play background music
    if not st.session_state.audio_playing:
        autoplay_audio("sounds/adventure_start.mp3")
        st.session_state.audio_playing = True

    tab1, tab2, tab3, tab4 = st.tabs(["Game", "Agentic Story", "Party", "Inventory"])

    with tab1:
        display_game_controls()
        display_adventure_log()
        
        st.subheader("Story Progression")
        
        # Display the current narrative
        if st.session_state.story_state["messages"]:
            st.write("Current story state:")
            st.write(st.session_state.story_state["messages"][-1].content)
        else:
            st.write("No story has been started yet.")
        
        # Collect user input
        user_input = st.text_input("Enter your choice:", key="main_user_input", value=st.session_state.main_user_input)
        
        if st.button("Start/Advance Story", key="advance_story_button_tab1"):
            st.write("Button clicked!")
            current_state = st.session_state.story_state.copy()
            st.write(f"Current state before processing: {current_state}")
            
            try:
                if not current_state["messages"]:
                    st.write("Starting new story...")
                    # If it's the start of the story, we don't need user input
                    new_state = st.session_state.story_app(current_state)
                elif user_input:
                    st.write(f"Advancing story with user input: {user_input}")
                    # For subsequent turns, include user input
                    current_state["user_input"] = user_input
                    new_state = st.session_state.story_app(current_state)
                else:
                    st.warning("Please enter your choice before advancing the story.")
                    return

                st.write(f"New state after processing: {new_state}")
                
                if new_state == END:
                    st.write("The story has ended.")
                elif new_state.get("messages"):
                    st.session_state.story_state = new_state
                    st.write("New story content:")
                    st.write(st.session_state.story_state["messages"][-1].content)
                else:
                    st.write("No new content generated.")
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
            
            # Clear the input field after processing
            st.session_state.main_user_input = ""

    with tab2:
        st.header("🎭 Agentic Story")
        
        # Display the current narrative
        if st.session_state.story_state["messages"]:
            st.write(st.session_state.story_state["messages"][-1].content)
        
        # Collect user input
        user_input = st.text_input("Enter your choice:", key="tab2_user_input")
        
        if st.button("Advance Story"):
            if user_input:
                # Invoke the story graph with user input
                st.session_state.story_state = st.session_state.story_app.invoke(st.session_state.story_state, {"user_input": user_input})
                if st.session_state.story_state.get("messages") == END:
                    st.write("The story has ended.")
                else:
                    st.write(st.session_state.story_state["messages"][-1].content)
            else:
                st.warning("Please enter your choice before advancing the story.")

    with tab3:
        display_party_information()

    with tab4:
        display_inventory()

    if not check_openai_api():
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

def display_party_information():
    if st.session_state.party:
        st.header("🦸‍♀️ Party Information")
        for name, info in st.session_state.party.items():
            with st.expander(f"⚔️ {name}"):
                st.write(info)

def display_adventure_log():
    if st.session_state.game_state is not None:
        st.header("📜 Adventure Log")
        
        # Display the current scene image
        if "current_image_path" in st.session_state.game_state:
            image_path = st.session_state.game_state["current_image_path"]
            if image_path and os.path.exists(image_path):
                st.image(image_path, caption="Current Scene", use_column_width=True)
        
        for i, event in enumerate(st.session_state.game_state["story_progression"]):
            with st.container():
                st.markdown(f"<div style='background-color: #2a2a4e; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>{event}</div>", unsafe_allow_html=True)

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

                # Get a new image for the current scene
                new_image_path = get_image_for_scene(dm_response)
                st.session_state.game_state["current_image_path"] = new_image_path

                current_step += 1
                progress_bar.progress(1.0)

                st.session_state.game_state["turn"] += 1
                st.session_state.game_state["turn_participation"] = {name: False for name in st.session_state.game_state["turn_participation"]}

                if st.session_state.game_state["turn"] > TURN_LIMIT:
                    st.warning(f"The adventure has reached its end after {TURN_LIMIT} turns. Start a new game to continue playing!")

            progress_bar.empty()
            play_sound("turn_complete")
            st.rerun()
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
    
    difficulty = st.select_slider("Difficulty", options=["Easy", "Medium", "Hard"], value="Medium")

    if st.button("🗺️ New Adventure"):
        with st.spinner("Preparing an epic quest..."):
            st.session_state.game_state, _ = start_new_adventure(st.session_state.party, difficulty)
            
            # Generate music using Suno API
            music_prompt = "Create an epic fantasy adventure theme"
            music_url = generate_music_with_suno(music_prompt)
            
            if music_url:
                autoplay_audio(music_url)
            else:
                autoplay_audio("sounds/adventure_start.mp3")
            
            # Reset audio playing state
            st.session_state.audio_playing = True
            st.rerun()
            
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

def initialize_session_state():
    if 'game_state' not in st.session_state:
        st.session_state.game_state = None
    if 'party' not in st.session_state:
        st.session_state.party = {
            "Player 1": "Name: Elara Moonwhisper\nRace: Elf\nClass: Ranger\nBackstory: Raised in the enchanted forests of Silverdale, Elara is a skilled archer and nature expert.",
            "Player 2": "Name: Thorgar Ironheart\nRace: Dwarf\nClass: Warrior\nBackstory: A stalwart defender from the mountain stronghold of Khazad-Dûm, Thorgar wields his ancestral warhammer with pride.",
            "Player 3": "Name: Zephyr Shadowstep\nRace: Halfling\nClass: Rogue\nBackstory: Once a street urchin in the bustling port city of Freeport, Zephyr's quick wit and quicker hands have gotten him out of (and into) many scrapes.",
            "Player 4": "Name: Lyra Flameheart\nRace: Human\nClass: Mage\nBackstory: A prodigy from the Arcane Academy of Stormhaven, Lyra's thirst for magical knowledge is matched only by her fiery temper."
        }
    if 'audio_playing' not in st.session_state:
        st.session_state.audio_playing = False
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = initialize_rag()
    if 'story_app' not in st.session_state:
        st.session_state.story_app = initialize_story_graph()
    if 'story_state' not in st.session_state:
        st.session_state.story_state = {
            "messages": [],
            "characters": [],
            "current_scene": "start",
            "story_summary": ""
        }

def check_openai_api():
    if not check_openai_availability():
        st.error(f"OpenAI API or the model '{MODEL_NAME}' is not available. Please make sure you have a valid API key and the model is accessible.")
        st.info("To get an OpenAI API key, sign up for an account on the OpenAI website.")
        if st.button("Retry Connection"):
            st.experimental_rerun()
        return False
    return True

if __name__ == "__main__":
    main()