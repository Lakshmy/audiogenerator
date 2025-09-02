from datetime import datetime
import os
from crewai import Crew
from config import api_key, azure_endpoint, deployment_name, api_version, AZURE_SPEECH_KEY, AZURE_SPEECH_REGION, AZURE_SPEECH_LANG, AZURE_SPEECH_VOICE_1, AZURE_SPEECH_VOICE_2, OUTPUT_FILENAME, MIME_TYPES, global_llm
from agents import image_analyst, report_analyzer, podcast_writer, speech_synthesizer
from tasks import image_task, analysis_task, business_update_task, speech_task

if __name__ == "__main__":
    print("\nConfiguration values:")
    print(f"Endpoint: {azure_endpoint}")
    print(f"Deployment: {deployment_name}")
    print(f"API Key length: {len(api_key) if api_key else 0}")
    print(f"Api version: {api_version}")
    print("\nTesting Azure OpenAI connection...")
    try:
        llm = global_llm
        response = llm.invoke("Hello! This is a test message.")
        print("Connection successful!")
    except Exception as e:
        print(f"Error: {str(e)}")

    # Derive language automatically from the first voice for SSML tag
    try:
        if AZURE_SPEECH_VOICE_1 and isinstance(AZURE_SPEECH_VOICE_1, str) and AZURE_SPEECH_VOICE_1.startswith('<'):
            print(f"Warning: AZURE_SPEECH_VOICE_1 seems to contain a placeholder '{AZURE_SPEECH_VOICE_1}'. Using default language '{AZURE_SPEECH_LANG}'. Please replace the placeholder.")
        elif AZURE_SPEECH_VOICE_1 and isinstance(AZURE_SPEECH_VOICE_1, str):
            parts = AZURE_SPEECH_VOICE_1.split('-')
            if len(parts) >= 2:
                AZURE_SPEECH_LANG = f"{parts[0]}-{parts[1]}"
            else:
                print(f"Warning: Could not reliably determine language from voice '{AZURE_SPEECH_VOICE_1}'. Using default '{AZURE_SPEECH_LANG}'.")
        elif AZURE_SPEECH_VOICE_1 is None:
            print(f"Warning: AZURE_SPEECH_VOICE_1 is not set. Using default language '{AZURE_SPEECH_LANG}'.")
    except Exception as e:
        print(f"Warning: Error parsing voice name for language '{AZURE_SPEECH_VOICE_1}': {e}. Using default '{AZURE_SPEECH_LANG}'.")

    timestamp = int(datetime.now().timestamp())
    OUTPUT_FILENAME = f"podcast_{timestamp}.wav"
    print(f"Azure Speech Key: {AZURE_SPEECH_KEY}")
    print(f"Azure Speech Region: {AZURE_SPEECH_REGION}")
    if not AZURE_SPEECH_KEY:
        print("Warning: AZURE_SPEECH_KEY environment variable not found or empty.")
    if not AZURE_SPEECH_REGION:
        print("Warning: AZURE_SPEECH_REGION environment variable not found or empty.")
    print(f"Host 1 Voice: {AZURE_SPEECH_VOICE_1}")
    print(f"Host 2 Voice: {AZURE_SPEECH_VOICE_2}")
    print(f"Derived SSML Language: {AZURE_SPEECH_LANG}")
    print(f"Output Base Filename: {OUTPUT_FILENAME}")
    print(f"----------------------------")
    output_dir = os.path.dirname(OUTPUT_FILENAME)
    if output_dir and not os.path.exists(output_dir):
        print(f"Warning: Output directory '{output_dir}' does not exist. Attempting to create it.")
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"Successfully created output directory: {output_dir}")
        except OSError as e:
            print(f"Error: Failed to create output directory '{output_dir}': {e}")
            raise
    print(f"\n--- Starting Crew Workflow ---")
    print(f"Image Path: pbi_image.jpg")
    print(f"Output Audio Path: {OUTPUT_FILENAME}")
    print(f"--> Derived SSML Language: {AZURE_SPEECH_LANG}")
    print(f"----------------------------\n")
    crew = Crew(
        agents=[image_analyst, report_analyzer, podcast_writer, speech_synthesizer],
        tasks=[image_task, analysis_task, business_update_task, speech_task],
        verbose=2
    )
    print("\n--- Kicking off Crew ---")
    result = crew.kickoff()
    print("\n--- Crew Workflow Finished ---")
    print("Final result from Crew:", result)
    print(f"\n--- Post-Run Verification ---")
    if os.path.exists(OUTPUT_FILENAME):
        print(f"Success: Output audio file found at: {OUTPUT_FILENAME}")
        print(f"File size: {os.path.getsize(OUTPUT_FILENAME)} bytes")
    else:
        print(f"Error: Output audio file NOT found at the expected path: {OUTPUT_FILENAME}")
        print("Please check the logs above for errors during the analysis, formatting or synthesis tasks.")
    print(f"---------------------------\n")
