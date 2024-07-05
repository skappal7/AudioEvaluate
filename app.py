import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
import time
from concurrent.futures import ThreadPoolExecutor

# Set up the page
st.set_page_config(page_title="AI-Powered Call Evaluation System", page_icon="📞", layout="wide")

# File path for CSV
CSV_FILE = "calls_data.csv"

# API setup
WHISPER_API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"
headers = {"Authorization": "Bearer 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"}

# File processing functions
def process_audio_chunk(chunk):
    response = requests.post(WHISPER_API_URL, headers=headers, data=chunk)
    if response.status_code == 200:
        result = response.json()
        return result.get("text", "")
    return ""

def transcribe_audio(audio_file):
    chunk_size = 1024 * 1024  # 1MB chunks
    transcription = ""
    with audio_file as file:
        with ThreadPoolExecutor() as executor:
            futures = []
            while chunk := file.read(chunk_size):
                futures.append(executor.submit(process_audio_chunk, chunk))
            for future in futures:
                transcription += future.result()
    return transcription

# Analysis functions (simplified for this example)
def analyze_call(transcript):
    # This is a placeholder. In a real scenario, you'd use more sophisticated NLP here.
    categories = ["Customer Service", "Technical Support", "Sales", "Billing"]
    return np.random.choice(categories)

def evaluate_call(transcript):
    # This is a placeholder. In a real scenario, you'd use more sophisticated NLP here.
    criteria = ["Problem Addressed", "Professional Tone", "Customer Connection", "Acknowledgment", "Understanding Shown", "Clear Communication"]
    return {criterion: np.random.choice([True, False]) for criterion in criteria}

# UI Components
def format_duration(seconds):
    minutes, seconds = divmod(seconds, 60)
    return f"{minutes:02d}:{seconds:02d}"

def render_call_table(df):
    # Prepare the data for display
    display_df = df.copy()
    display_df['Call'] = display_df.apply(lambda row: f"{row['Call ID']} {format_duration(row['Duration'])}", axis=1)
    display_df['Issues resolved?'] = display_df['Issues resolved?'].apply(lambda x: '█' * int(x * 10))
    
    # Reorder and rename columns
    columns = ['Call', 'Issues resolved?', 'Polite?', 'Rapport?', 'Apology?', 'Empathy?', 'Jargon-free?']
    display_df = display_df[columns]
    
    # Replace boolean values with symbols
    for col in ['Polite?', 'Rapport?', 'Apology?', 'Empathy?', 'Jargon-free?']:
        display_df[col] = display_df[col].map({True: '✅', False: '❌'})
    
    # Display the table
    st.table(display_df)

def show_call_details(df, call_id):
    call_data = df[df['Call ID'] == call_id].iloc[0]
    st.subheader(f"Details for {call_id}")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"Duration: {format_duration(call_data['Duration'])}")
        st.write(f"Category: {call_data['Category']}")
    with col2:
        for criterion in ['Issues resolved?', 'Polite?', 'Rapport?', 'Apology?', 'Empathy?', 'Jargon-free?']:
            st.write(f"{criterion} {'✅' if call_data[criterion] else '❌'}")
    st.text_area("Transcript", value=call_data['Transcript'], height=200)

# Main app
def main():
    st.title("AI-Powered Call Evaluation System")

    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
    else:
        df = pd.DataFrame(columns=['Call ID', 'Duration', 'Category', 'Transcript', 'Issues resolved?', 'Polite?', 'Rapport?', 'Apology?', 'Empathy?', 'Jargon-free?'])

    uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "flac"])

    if uploaded_file is not None:
        st.audio(uploaded_file)

        if st.button("Analyze Call"):
            with st.spinner("Transcribing and analyzing call..."):
                transcript = transcribe_audio(uploaded_file)
                category = analyze_call(transcript)
                evaluation = evaluate_call(transcript)

                new_row = pd.DataFrame({
                    'Call ID': [f"call-{len(df)+1:03d}"],
                    'Duration': [300],  # Placeholder duration
                    'Category': [category],
                    'Transcript': [transcript],
                    'Issues resolved?': [np.random.random()],  # Placeholder for demonstration
                    'Polite?': [evaluation['Professional Tone']],
                    'Rapport?': [evaluation['Customer Connection']],
                    'Apology?': [evaluation['Acknowledgment']],
                    'Empathy?': [evaluation['Understanding Shown']],
                    'Jargon-free?': [evaluation['Clear Communication']]
                })

                df = pd.concat([df, new_row], ignore_index=True)
                df.to_csv(CSV_FILE, index=False)
                st.success("Call analyzed and added to the dataset!")

    if not df.empty:
        render_call_table(df)

        # Navigation
        col1, col2, col3 = st.columns([1,3,1])
        with col1:
            if st.button("⬅️ Previous"):
                st.session_state.current_call_index = max(0, st.session_state.get('current_call_index', 0) - 1)
        with col3:
            if st.button("Next ➡️"):
                st.session_state.current_call_index = min(len(df) - 1, st.session_state.get('current_call_index', 0) + 1)

        # Show call details
        current_call = df.iloc[st.session_state.get('current_call_index', 0)]
        show_call_details(df, current_call['Call ID'])

        # Download CSV
        st.download_button(
            label="Download data as CSV",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name="call_evaluation_data.csv",
            mime="text/csv",
        )
    else:
        st.info("No calls analyzed yet. Upload and analyze a call to see results.")

if __name__ == "__main__":
    main()
