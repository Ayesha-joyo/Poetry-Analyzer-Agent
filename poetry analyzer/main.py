from agents import Agent, Runner, trace, function_tool
from connection import config
import asyncio
from dotenv import load_dotenv

load_dotenv()



lyric_poetry_agent = Agent(
    name="Lyric Poetry Agent",
    instructions=""" You are a Lyric Poetry Agent. Your specialty is analyzing lyric poems, which express personal thoughts, emotions, and feelings.

When a poem is routed to you:
- Identify the main emotion(s) expressed (e.g., love, sadness, hope).
Do NOT classify the poem. ONLY analyze its emotional content.
""",
)


daramatic_poetry_agent = Agent(
    name="Daramatic Poetry Agent",
    instructions="""
You are a Dramatic Poetry Agent. Your job is to analyze poems written to be performed like a theatrical monologue or dialogue.

When you receive a poem:
- Describe the dramatic situation — what is happening on stage?

Do NOT tell the story yourself — only analyze the dramatic qualities.

""",
    model="gpt-3.5-turbo"  
)

narrative_poetry_agent = Agent(
    name="Narrative Poetry Agent",
    instructions="""
                You are a Narrative Poetry Agent. Your task is to analyze poems that tell a story.

When a poem is routed to you:
- Summarize the basic storyline (characters, setting, problem, outcome).



""",
    model="gpt-3.5-turbo" 
)

parent_agent = Agent(
    name="Parent Agent",
    instructions="""
       You are a parent (orchestrator) agent. Your job is to intelligently route a given poem (input text) to the correct poetry analyst agent based on its type.

Routing Rules:
- If the poem is expressing personal emotions or inner thoughts (like joy, grief, love) → send to the Lyric Poetry Agent.
- If the poem tells a story with characters, setting, and sequence of events → send to the Narrative Poetry Agent.
- If the poem looks like a speech/monologue meant to be performed or acted (as in theater) → send to the Dramatic Poetry Agent.

If the input clearly shows mixed traits, choose the **strongest dominant style**. If it cannot be identified, politely explain you do not know the type.
You do NOT analyze poems yourself — only delegate.

    """,
    handoffs=[lyric_poetry_agent, daramatic_poetry_agent, narrative_poetry_agent],
    
)

async def main():
    with trace("Class 06"):
        result = await Runner.run(
            parent_agent,
            """
                Why do you stare, O silent crowd before me?
Have I not bled my truth upon this stage?
Must I kneel yet again and beg for mercy,
While anger burns behind this fragile cage?

Speak then, judge my cries or grant me pardon,
For I have loved too boldly, sinned too loud.
Let this confession echo through your garden,
And let my name not perish in the crowd.
            """,
            run_config=config
        )
        print("Final Output:", result.final_output)
        print("Last Agent:", result.last_agent.name)

if __name__ == "__main__":
    asyncio.run(main())
