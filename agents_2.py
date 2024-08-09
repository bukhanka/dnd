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

# Update the choose_topic function
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

# Modify the StoryState to include the current educational topic
class StoryState(TypedDict):
    # ... existing fields ...
    current_educational_topic: str

# Update the initial state
story_state = {
    # ... existing fields ...
    "current_educational_topic": ""
}