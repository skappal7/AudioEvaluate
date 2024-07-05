import streamlit as st
import pandas as pd
import speech_recognition as sr
from textblob import TextBlob
import spacy

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Initialize data storage
data = []

def transcribe_audio(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
    transcript = recognizer.recognize_google(audio)
    return transcript

def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

def extract_keywords(text):
    doc = nlp(text)
    return [chunk.text for chunk in doc.noun_chunks]

st.title("Voice Transcript Analyzer")

uploaded_files = st.file_uploader("Upload audio files", type=["wav"], accept_multiple_files=True)

if uploaded_files:
    for audio_file in uploaded_files:
        with st.spinner(f"Transcribing {audio_file.name}..."):
            transcript = transcribe_audio(audio_file)
            st.subheader(f"Transcript for {audio_file.name}")
            st.write(transcript)

            with st.spinner("Analyzing sentiment..."):
                sentiment = analyze_sentiment(transcript)

            with st.spinner("Extracting keywords..."):
                keywords = extract_keywords(transcript)

            st.audio(audio_file, format='audio/wav')

            # User inputs for parameters
            st.write("Please evaluate the following parameters:")
            col1, col2, col3 = st.columns(3)
            with col1:
                issues_resolved = st.checkbox("Issues resolved?", key=f"resolved_{audio_file.name}")
                polite = st.checkbox("Polite?", key=f"polite_{audio_file.name}")
                rapport = st.checkbox("Rapport?", key=f"rapport_{audio_file.name}")
            with col2:
                apology = st.checkbox("Apology?", key=f"apology_{audio_file.name}")
                empathy = st.checkbox("Empathy?", key=f"empathy_{audio_file.name}")
                jargon_free = st.checkbox("Jargon-free?", key=f"jargon_{audio_file.name}")

            # Save the data
            data.append({
                "Call": audio_file.name,
                "Issues resolved?": issues_resolved,
                "Polite?": polite,
                "Rapport?": rapport,
                "Apology?": apology,
                "Empathy?": empathy,
                "Jargon-free?": jargon_free,
                "Transcript": transcript,
                "Sentiment Score": sentiment,
                "Keywords": ", ".join(keywords)
            })

    # Display the results in a table
    if data:
        df = pd.DataFrame(data)
        st.subheader("Call Analysis Summary")

        # Style the DataFrame to look similar to the provided table
        def highlight_cells(val):
            color = 'green' if val else 'red'
            return f'background-color: {color}'

        def highlight_text(val):
            if isinstance(val, bool):
                return '✔️' if val else '❌'
            return val

        styled_df = df.style.applymap(highlight_cells, subset=['Issues resolved?', 'Polite?', 'Rapport?', 'Apology?', 'Empathy?', 'Jargon-free?'])
        styled_df = styled_df.format(highlight_text)
        
        st.dataframe(styled_df, height=500)

        # Option to download the data as CSV
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", data=csv, file_name="call_analysis.csv", mime="text/csv")
