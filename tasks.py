from crewai import Task
from agents import image_analyst, report_analyzer, podcast_writer, speech_synthesizer
from config import OUTPUT_FILENAME

LOCAL_IMAGE_PATH = "pbi_image.jpg"

image_task = Task(
    description=(
        f"Analyze the image located at {LOCAL_IMAGE_PATH}. Provide a comprehensive and objective textual description "
        "detailing all visible elements, scenes, objects, text, charts, graphs, and any discernible context or data presented visually. "
        "This description will be used by the Business Report Analyzer."
    ),
    agent=image_analyst,
    expected_output="A detailed, objective textual description of the image's contents, focusing on data and textual elements."
)

analysis_task = Task(
    description=(
        "Review the detailed image description provided in the context. "
        "Based *solely* on the information presented in that description (text, data points, chart descriptions etc.), "
        "perform a thorough business analysis. Identify key insights, significant trends, important figures, and potential implications. "
        "Structure your findings as a written essay. Start with a clear 'Executive Summary' section summarizing the main points, "
        "followed by a more detailed 'Analysis' section elaborating on the findings. "
        "This report will be used to create a podcast script."
    ),
    agent=report_analyzer,
    expected_output=(
        "A well-structured written analysis in essay format. "
        "It MUST begin with an 'Executive Summary' section. "
        "It MUST be followed by a detailed 'Analysis' section. "
        "The analysis must be based ONLY on the information from the input image description."
    ),
    context=[image_task]
)

business_update_task = Task(
    description=(
        "Okay, I have the business analysis report here (check the context). My task is to translate the key findings and insights from this report "
        "into a natural, monologue for a podcast featuring the host. The conversation should flow well and make the analysis easy for listeners to grasp. "
        "\n\n**Formatting Guide for Audio:**\n"
        f"*   **Host 1:** Use the `...` tag for everything Host 1 says.\n"
        "*   **Structure:** Please alternate between these tags as the hosts speak. Getting these voice tags right is crucial for the audio step!\n"
        "*   **Wrapping (Optional but helpful):** If you can, please also wrap the entire dialogue within the main `...` tags.\n"
        "*   **Content:** Just include the words the hosts will actually say. No extra labels like 'Host 1:', notes about music, or markdown formatting around the SSML itself."
    ),
    agent=podcast_writer,
    expected_output=(
        "A complete podcast script formatted as monologue. It should feature alternating "
        f"`...` and `...` tags "
        "containing the spoken lines for the host, accurately reflecting the input analysis in a monologue style. "
        "Ideally, the whole script will be enclosed in `...` tags."
    ),
    context=[analysis_task]
)

speech_task = Task(
    description=(
        "Process the podcast monologue script received from the previous task (available in context). "
        "Step 1: Use the 'SSML Formatter Tool' to ensure the script is wrapped in valid `...` tags with the correct language attribute (derived automatically from config). "
        "Step 2: Take the **output** from the 'SSML Formatter Tool' and use the 'SSML to Speech Tool' to synthesize this final, validated SSML into a single audio track. "
        f"Ensure the final audio is saved correctly as a .wav file to the path '{OUTPUT_FILENAME}'. "
        "Report success or failure, including the output path on success or detailed error messages on failure."
    ),
    agent=speech_synthesizer,
    expected_output=(
        "Confirmation of successful audio file generation from the formatted SSML input, including the "
        f"file path where the .wav file was saved (expected: {OUTPUT_FILENAME}), or a descriptive error message if any step failed."
    ),
    context=[business_update_task]
)
