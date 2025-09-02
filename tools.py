import os
import base64
import time
import re
from pathlib import Path
from langchain.schema import HumanMessage, AIMessage
from azure.cognitiveservices.speech import SpeechConfig, SpeechSynthesizer, AudioConfig, ResultReason, CancellationReason, SpeechSynthesisOutputFormat
from config import global_llm, AZURE_SPEECH_KEY, AZURE_SPEECH_REGION, OUTPUT_FILENAME, MIME_TYPES, AZURE_SPEECH_LANG

def describe_image(local_image_path: str) -> str:
    try:
        llm = global_llm
        if llm is None:
            return "Error: LLM not provided to the tool. Make sure to set global_llm before using this tool."
        image_path = Path(local_image_path)
        if not image_path.exists():
            return f"Error: Image file not found at {image_path}"
        with open(image_path, "rb") as img_file:
            img_data = base64.b64encode(img_file.read()).decode("utf-8")
        extension = image_path.suffix.lower()
        mime_type = MIME_TYPES.get(extension, "application/octet-stream")
        image_content = f"data:{mime_type};base64,{img_data}"
        message = HumanMessage(
            content=[
                {"type": "text", "text": "Please describe this image in detail, focusing on any text, charts, graphs, or data presented:"},
                {"type": "image_url", "image_url": {"url": image_content}}
            ]
        )
        response = llm.invoke([message])
        if isinstance(response, AIMessage):
            return response.content
        else:
            return str(response)
    except Exception as e:
        return f"Error processing image: {str(e)}"

def ssml_to_speech(ssml_input: str) -> str:
    try:
        speech_config = SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
        speech_config.set_speech_synthesis_output_format(SpeechSynthesisOutputFormat.Riff24Khz16BitMonoPcm)
        ssml_content = str(ssml_input)
        if not ssml_content:
            return "Error: No SSML content provided for speech synthesis."
        output_dir = os.path.dirname(OUTPUT_FILENAME)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        audio_config = AudioConfig(filename=OUTPUT_FILENAME)
        synthesizer = SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
        result = synthesizer.speak_ssml_async(ssml_content).get()
        del synthesizer; del audio_config; time.sleep(0.1)
        if result.reason == ResultReason.SynthesizingAudioCompleted:
            wait_time = 0
            while not (os.path.exists(OUTPUT_FILENAME) and os.path.getsize(OUTPUT_FILENAME) > 0) and wait_time < 5:
                time.sleep(0.5); wait_time += 0.5
            if not (os.path.exists(OUTPUT_FILENAME) and os.path.getsize(OUTPUT_FILENAME) > 0):
                return f"Error: Output file {OUTPUT_FILENAME} is missing or empty after synthesis completed."
            else:
                return f"Speech synthesis successful. File saved at: {OUTPUT_FILENAME}"
        elif result.reason == ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            error_message = f"SSML synthesis canceled. Reason: {cancellation_details.reason}"
            if cancellation_details.reason == CancellationReason.Error:
                error_message += f" Error details: {cancellation_details.error_details}"
            if os.path.exists(OUTPUT_FILENAME):
                try: os.remove(OUTPUT_FILENAME)
                except OSError: pass
            return error_message
        else:
            error_message = f"Unexpected SSML synthesis result: {result.reason}"
            if os.path.exists(OUTPUT_FILENAME):
                try: os.remove(OUTPUT_FILENAME)
                except OSError: pass
            return error_message
    except Exception as e:
        if os.path.exists(OUTPUT_FILENAME):
            try: os.remove(OUTPUT_FILENAME)
            except OSError: pass
        return f"Error during SSML to speech processing: {str(e)}"

def format_ssml_tool(dialogue_content: str) -> str:
    ssml_body = str(dialogue_content).strip()
    ssml_body = re.sub(r'^```xml\s*', '', ssml_body, flags=re.IGNORECASE | re.MULTILINE)
    ssml_body = re.sub(r'\s*```$', '', ssml_body, flags=re.MULTILINE)
    ssml_body = ssml_body.strip()
    lang_attribute = f'xml:lang="{AZURE_SPEECH_LANG}"'
    is_already_wrapped = ssml_body.startswith('<speak') and ssml_body.endswith('</speak>')
    has_correct_lang = lang_attribute in ssml_body[:150]
    if is_already_wrapped and has_correct_lang:
        return ssml_body
    elif is_already_wrapped and not has_correct_lang:
        import re
        pattern = re.compile(r'(<speak[^>]*)(\s*xml:lang="[^"]*")?([^>]*>)', re.IGNORECASE)
        if pattern.search(ssml_body):
            ssml_body = pattern.sub(rf'\1 {lang_attribute}\3', ssml_body, count=1)
        else:
            ssml_body = ssml_body.replace('<speak', f'<speak {lang_attribute}', 1)
        return ssml_body
    else:
        ssml_full = f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" {lang_attribute}>\n{ssml_body}\n</speak>'
        return ssml_full
