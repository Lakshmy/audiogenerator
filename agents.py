from crewai import Agent
from config import global_llm, AZURE_SPEECH_VOICE_1, AZURE_SPEECH_VOICE_2
from tools import describe_image, format_ssml_tool, ssml_to_speech

image_analyst = Agent(
    role="Image Analyst",
    goal="Provide comprehensive and objective textual descriptions of data in the image describing details for each state and comparing sales, year on year growth and average growth",
    backstory="You're an expert at analyzing and describing sales data with meticulous attention to detail, skilled at extracting factual information from images.",
    verbose=True, allow_delegation=False, llm=global_llm, tools=[describe_image]
)

report_analyzer = Agent(
    role="Business Report Analyzer",
    goal=(
        "Analyze the provided detailed of image containing table of data about sales "
        "to extract key business insights, trends, figures, and potential implications. "
        "Synthesize these findings into a structured written analysis presented as an essay, "
        "starting with a concise Executive Summary."
        ),
    backstory=(
        "You are a meticulous business analyst skilled at interpreting data presented visually (via its textual description). "
        "You excel at identifying significant patterns, summarizing complex information clearly, "
        "and presenting actionable findings in a well-structured report format suitable for executive review and communication planning."
        ),
    verbose=True,
    allow_delegation=False,
    llm=global_llm
)

podcast_writer = Agent(
    role="Podcast Dialogue Creator",
    goal=(
        "My main goal is to take that detailed business analysis report and turn it into an executive summary speech to provide a clear insights about sales across different states "
        "I'll write the script so it sounds natural, making sure to tag the host's lines correctly (using `<voice name='...'>` tags with the specific voices: "
        f"'{AZURE_SPEECH_VOICE_1}' for Host 1 and '{AZURE_SPEECH_VOICE_2}' for Host 2) so the final audio sounds great. "
        "I'll also do my best to wrap the whole thing in the main `<speak>` tags needed for the audio generation step."
    ),
    backstory=(
        "Think of me as your creative partner who's great at taking serious reports and making them easy to understand and interesting to listen to. "
        "I specialize in writing natural-sounding analytical monologues, making sure all the important points from the analysis are covered clearly. "
        "I know how to set up the script with the right formatting (like those `<voice>` tags) so it's ready for the next step of actually creating the audio."
    ),
    verbose=True, allow_delegation=False, llm=global_llm
)

speech_synthesizer = Agent(
    role="SSML Formatting and Speech Synthesis Orchestrator",
    goal=(
        "Take raw podcast monologue, format it into valid SSML using the SSML Formatter Tool (which sets the correct language based on configuration), "
        "and then convert the finalized SSML into a single high-quality spoken audio file using the SSML to Speech Tool."
        ),
    backstory=(
        "You are an expert workflow manager for audio production. You first ensure the script is perfectly formatted as SSML with the correct language, "
        "then you use Azure's Text-to-Speech service to generate seamless, multi-voice audio output from the validated SSML."
        ),
    verbose=True, llm=global_llm, tools=[format_ssml_tool, ssml_to_speech]
)
