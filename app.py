import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from transformers import pipeline
import requests
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
import time

# Set up the page
st.set_page_config(page_title="AI-Powered Call Evaluation System", page_icon="📞", layout="wide")

# File path for CSV
CSV_FILE = "calls_data.csv"

# API setup
WHISPER_API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"
headers = {"Authorization": "Bearer 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"}

# Load AI models
@st.cache_resource
def load_models():
    text_classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
    zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    return text_classifier, zero_shot_classifier

text_classifier, zero_shot_classifier = load_models()

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

# Analysis functions
def analyze_call(transcript):
    categories = ["Customer Service", "Technical Support", "Sales", "Billing"]
    result = zero_shot_classifier(transcript, categories)
    return result["labels"][0]

def evaluate_call(transcript):
    criteria = ["Problem Addressed", "Professional Tone", "Customer Connection", "Acknowledgment", "Understanding Shown", "Clear Communication"]
    results = {}
    for criterion in criteria:
        result = zero_shot_classifier(transcript, [criterion, f"Not {criterion}"])
        results[criterion] = result["scores"][0] > 0.5
    return results

# UI Components
def render_call_table(df):
    st.markdown("""
    <style>
    .call-table {
        font-family: Arial, sans-serif;
        border-collapse: collapse;
        width: 100%;
    }
    .call-table td, .call-table th {
        border: 1px solid #ddd;
        padding: 8px;
    }
    .call-table tr:nth-child(even) {background-color: #f2f2f2;}
    .call-table tr:hover {background-color: #ddd; cursor: pointer;}
    .call-table th {
        padding-top: 12px;
        padding-bottom: 12px;
        text-align: left;
        background-color: #4CAF50;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

    table_html = """
    <table class="call-table">
        <tr>
            <th>#</th>
            <th>Call</th>
            <th>Issues resolved?</th>
            <th>Polite?</th>
            <th>Rapport?</th>
            <th>Apology?</th>
            <th>Empathy?</th>
            <th>Jargon-free?</th>
        </tr>
    """

    for _, row in df.iterrows():
        table_html += f"""
        <tr onclick="handleRowClick('{row['Call ID']}')">
            <td>{row.name + 1}</td>
            <td>{row['Call ID']} {row['Duration']}</td>
            <td><div style="width:100%;background-color:{'green' if row['Issues resolved?'] else 'red'};height:20px;"></div></td>
            <td>{'✅' if row['Polite?'] else '❌'}</td>
            <td>{'✅' if row['Rapport?'] else '❌'}</td>
            <td>{'✅' if row['Apology?'] else '❌'}</td>
            <td>{'✅' if row['Empathy?'] else '❌'}</td>
            <td>{'✅' if row['Jargon-free?'] else '❌'}</td>
        </tr>
        """

    table_html += "</table>"

    st.markdown(table_html, unsafe_allow_html=True)

    st.markdown("""
    <script>
    function handleRowClick(callId) {
        Streamlit.setComponentValue(callId);
    }
    </script>
    """, unsafe_allow_html=True)

def show_call_details(df, call_id):
    call_data = df[df['Call ID'] == call_id].iloc[0]
    st.subheader(f"Details for {call_id}")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"Duration: {call_data['Duration']} seconds")
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
                    'Call ID': [f"Call-{int(time.time())}"],
                    'Duration': [300],  # Placeholder duration
                    'Category': [category],
                    'Transcript': [transcript],
                    'Issues resolved?': [evaluation['Problem Addressed']],
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
