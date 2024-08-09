from typing import TypedDict, List, Annotated, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import ToolNode
import operator
from openai import OpenAI  # Add OpenAI import
import os

# Define our state
class StoryState(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage], operator.add]
    characters: List[dict]
    current_scene: str
    story_summary: str  # Add story summary to keep track of the overall narrative
    educational_topics: Dict[str, Any]
    player_knowledge: Dict[str, float]
    current_educational_topic: str

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY')) 

# Improved functions for our nodes
def dungeon_master(state: StoryState) -> StoryState:
    # Use gpt-4o-mini to generate a more engaging narrative
    prompt = f"Current scene: {state['current_scene']}\nStory summary: {state['story_summary']}\nIncorporate educational content about {state['current_educational_topic']}. Current player knowledge: {state['player_knowledge']}\nGenerate an engaging narrative and present choices to the player."
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are a creative and engaging dungeon master."},
                  {"role": "user", "content": prompt}]
    )
    narrative = response.choices[0].message.content
    return {"messages": [AIMessage(content=narrative)]}

def user_choice(state: StoryState) -> StoryState:
    print(f"\nCurrent narrative: {state['messages'][-1].content}")
    choice = input("Enter your choice: ")
    if "educational_challenge" in state['messages'][-1].content:
        answer = input("Your answer: ")
        state['player_knowledge'] = update_knowledge(state['player_knowledge'], answer)
    return {"messages": [HumanMessage(content=choice)]}

def story_manager(state: StoryState) -> StoryState:
    # Use gpt-4o-mini to update the story arc and determine the next scene
    prompt = f"Current scene: {state['current_scene']}\nUser choice: {state['messages'][-1].content}\nUpdate the story arc and determine the next scene."
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are a creative story manager for an interactive narrative."},
                  {"role": "user", "content": prompt}]
    )
    result = response.choices[0].message.content
    next_scene, summary_update = result.split('\n', 1)
    
    # Choose an educational topic based on the next scene
    chosen_topic = choose_topic(state['educational_topics'], state['player_knowledge'], next_scene)
    
    return {
        "current_scene": next_scene,
        "story_summary": state["story_summary"] + " " + summary_update,
        "messages": [AIMessage(content=f"The story moves to: {next_scene}")],
        "current_educational_topic": chosen_topic
    }

def character_creation(state: StoryState) -> StoryState:
    # Use gpt-4o-mini to create more interesting and context-appropriate characters
    prompt = f"Current scene: {state['current_scene']}\nStory summary: {state['story_summary']}\nCreate a new character that fits the current narrative."
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are a creative character designer for an interactive story."},
                  {"role": "user", "content": prompt}]
    )
    new_character = response.choices[0].message.content
    return {"characters": state["characters"] + [{"description": new_character}]}

def educational_challenge(state: StoryState) -> StoryState:
    topic = choose_topic(state['educational_topics'], state['player_knowledge'])
    prompt = f"Create an educational challenge about {topic} that fits the current narrative: {state['current_scene']}"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are an educational game designer."},
                  {"role": "user", "content": prompt}]
    )
    challenge = response.choices[0].message.content
    return {"messages": [AIMessage(content=challenge)]}

def choose_topic(topics: Dict[str, Any], knowledge: Dict[str, float], scene: str) -> str:
    # Logic to choose a topic based on current knowledge levels and the upcoming scene
    prompt = f"Given the upcoming scene: '{scene}', and the available topics: {topics}, choose the most appropriate educational topic. Consider the player's current knowledge levels: {knowledge}."
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are an educational content curator for an interactive story."},
                  {"role": "user", "content": prompt}]
    )
    chosen_topic = response.choices[0].message.content.strip()
    return chosen_topic

def update_knowledge(knowledge: Dict[str, float], answer: str) -> Dict[str, float]:
    # Logic to update player's knowledge based on their answer
    pass

# Create our graph
story_graph = StateGraph(StoryState)

# Add nodes
story_graph.add_node("dungeon_master", dungeon_master)
story_graph.add_node("user_choice", user_choice)
story_graph.add_node("story_manager", story_manager)
story_graph.add_node("character_creation", character_creation)
story_graph.add_node("educational_challenge", educational_challenge)

# Add edges
story_graph.add_edge("dungeon_master", "user_choice")
story_graph.add_edge("user_choice", "story_manager")
story_graph.add_edge("story_manager", "dungeon_master")
story_graph.add_edge("story_manager", "character_creation")
story_graph.add_edge("character_creation", "dungeon_master")
story_graph.add_edge("story_manager", "educational_challenge")
story_graph.add_edge("educational_challenge", "user_choice")

# Set entry point
story_graph.set_entry_point("dungeon_master")

# Modify the should_end_story function to use gpt-4o-mini for more dynamic story ending conditions
def should_end_story(state: StoryState) -> bool:
    prompt = f"Story summary: {state['story_summary']}\nCurrent scene: {state['current_scene']}\nDetermine if the story should end based on narrative progression and engagement."
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are an expert storyteller determining if a story should conclude."},
                  {"role": "user", "content": prompt}]
    )
    should_end = response.choices[0].message.content.lower().startswith('yes')
    return should_end

story_graph.add_conditional_edges(
    "story_manager",
    should_end_story,
    {True: END, False: "dungeon_master"}
)

# Compile the graph
story_app = story_graph.compile()

# Run the story
story_state = {
    "messages": [],
    "characters": [],
    "current_scene": "start",
    "story_summary": "",
    "educational_topics": {
        "history": ["Ancient civilizations", "World Wars", "Industrial Revolution"],
        "science": ["Basic physics", "Chemistry fundamentals", "Biology concepts"],
        "literature": ["Shakespeare", "World mythology", "Classic novels"]
    },
    "player_knowledge": {
        "history": 0.0,
        "science": 0.0,
        "literature": 0.0
    },
    "current_educational_topic": ""
}

while True:
    story_state = story_app.invoke(story_state)
    if story_state.get("messages") == END:
        break

print("\nStory ended. Final state:")
print(story_state)