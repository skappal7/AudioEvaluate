import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import whisper
import torch
from transformers import pipeline
import os
import tempfile

# Set up AI models
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

@st.cache_resource
def load_text_classification_model():
    return pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

@st.cache_resource
def load_zero_shot_classification_model():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

whisper_model = load_whisper_model()
text_classifier = load_text_classification_model()
zero_shot_classifier = load_zero_shot_classification_model()

# Function to transcribe audio
def transcribe_audio(audio_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_file.read())
        temp_audio_path = temp_audio.name

    result = whisper_model.transcribe(temp_audio_path)
    os.unlink(temp_audio_path)
    return result["text"]

# Function to analyze call category
def analyze_call(transcript):
    categories = ["Customer Service", "Technical Support", "Sales", "Billing"]
    result = zero_shot_classifier(transcript, categories)
    return result["labels"][0]

# Function to evaluate call
def evaluate_call(transcript):
    criteria = [
        "Problem Addressed?",
        "Professional Tone?",
        "Customer Connection?",
        "Acknowledgment?",
        "Understanding Shown?",
        "Clear Communication?"
    ]
    results = {}
    for criterion in criteria:
        result = zero_shot_classifier(transcript, [criterion, f"Not {criterion}"])
        results[criterion] = result["labels"][0] == criterion
    return results

# Function to load or create data
@st.cache_data
def load_data():
    if os.path.exists("call_data.csv"):
        return pd.read_csv("call_data.csv")
    else:
        return pd.DataFrame(columns=["Call ID", "Duration", "Category", "Transcript"] + [
            "Problem Addressed?",
            "Professional Tone?",
            "Customer Connection?",
            "Acknowledgment?",
            "Understanding Shown?",
            "Clear Communication?"
        ])

# Main app
def main():
    st.title("AI-Powered Call Evaluation System")

    # Load data
    df = load_data()

    # File uploader
    uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")

        if st.button("Analyze Call"):
            with st.spinner("Transcribing and analyzing call..."):
                # Transcribe audio
                transcript = transcribe_audio(uploaded_file)

                # Analyze call
                category = analyze_call(transcript)

                # Evaluate call
                evaluation = evaluate_call(transcript)

                # Add to dataframe
                new_row = {
                    "Call ID": f"Call-{len(df)+1:03d}",
                    "Duration": 300,  # Placeholder, replace with actual duration
                    "Category": category,
                    "Transcript": transcript,
                    **evaluation
                }
                df = df.append(new_row, ignore_index=True)

                # Save updated dataframe
                df.to_csv("call_data.csv", index=False)

                st.success("Call analyzed and added to the database!")

    # Display data
    st.subheader("Call Evaluation Results")
    st.dataframe(df)

    # Call details
    st.subheader("Call Details")
    selected_call = st.selectbox("Select a call to view details", df["Call ID"])
    call_data = df[df["Call ID"] == selected_call].iloc[0]

    col1, col2 = st.columns(2)
    with col1:
        st.write(f"Call ID: {call_data['Call ID']}")
        st.write(f"Duration: {call_data['Duration']} seconds")
        st.write(f"Category: {call_data['Category']}")

    with col2:
        for criterion in [
            "Problem Addressed?",
            "Professional Tone?",
            "Customer Connection?",
            "Acknowledgment?",
            "Understanding Shown?",
            "Clear Communication?"
        ]:
            st.write(f"{criterion} {'✅' if call_data[criterion] else '❌'}")

    st.text_area("Transcript", value=call_data["Transcript"], height=200)

    # Data visualization
    st.subheader("Performance Overview")
    fig = go.Figure(data=[
        go.Bar(name=criterion, x=df["Call ID"], y=df[criterion])
        for criterion in [
            "Problem Addressed?",
            "Professional Tone?",
            "Customer Connection?",
            "Acknowledgment?",
            "Understanding Shown?",
            "Clear Communication?"
        ]
    ])
    fig.update_layout(barmode='group', height=400)
    st.plotly_chart(fig)

if __name__ == "__main__":
    main()
