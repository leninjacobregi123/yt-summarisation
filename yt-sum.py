import streamlit as st
import whisper
import torch
import os
import yt_dlp
from transformers import pipeline

# Check if a GPU is available and set the device accordingly
device = "cuda" if torch.cuda.is_available() else "cpu"
st.write(f"Running on: {device.upper()}")

# Load Whisper Large multilingual model
model = whisper.load_model("large", device=device)

# Load the summarization pipeline using Facebook's BART model on the correct device
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if torch.cuda.is_available() else -1)

# Define chunk length in seconds for transcription
chunk_len_s = 10

def download_video(youtube_url, audio_file_path):
    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': audio_file_path.replace('.mp3', '') + '.%(ext)s',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])

        final_audio_file_path = audio_file_path if audio_file_path.endswith('.mp3') else audio_file_path + '.mp3'
        st.success(f"Audio downloaded and saved as {final_audio_file_path}")
        return final_audio_file_path

    except Exception as e:
        st.error(f"Error downloading video: {e}")
        return None

def transcribe_audio_in_chunks(audio_file_path, chunk_len_s):
    try:
        if not os.path.exists(audio_file_path):
            st.error(f"Error: MP3 file {audio_file_path} not found")
            return None

        # Load and preprocess the audio file
        audio = whisper.load_audio(audio_file_path)
        audio_length = len(audio) / whisper.audio.SAMPLE_RATE

        # Transcribe the audio in chunks
        transcription = ""
        for start in range(0, int(audio_length), chunk_len_s):
            end = min(start + chunk_len_s, int(audio_length))
            chunk = audio[int(start * whisper.audio.SAMPLE_RATE):int(end * whisper.audio.SAMPLE_RATE)]
            chunk = whisper.pad_or_trim(chunk)
            result = model.transcribe(chunk, language="en")
            transcription += result['text'] + " "

        return transcription.strip()
    except Exception as e:
        st.error(f"Error transcribing audio: {e}")
        return None
    
def summarize_text(text):
    try:
        max_chunk_size = 1024
        text_chunks = [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]

        summaries = []
        for chunk in text_chunks:
            summary = summarizer(chunk, max_length=150, min_length=40, do_sample=False)[0]['summary_text']
            summaries.append(summary)

        combined_summary = " ".join(summaries)
        return combined_summary
    except Exception as e:
        st.error(f"Error summarizing text: {e}")
        return None

def main(youtube_url):
    audio_file_path = "audio.mp3"
    downloaded_audio_path = download_video(youtube_url, audio_file_path)
    
    if downloaded_audio_path:
        transcription = transcribe_audio_in_chunks(downloaded_audio_path, chunk_len_s)
        if transcription:
            st.write("Transcription complete")
            summary = summarize_text(transcription)
            if summary:
                return transcription, summary
    return None, None

# Streamlit UI
st.title("YouTube Video Transcription and Summarization")

# Input for YouTube video URL
youtube_url = st.text_input("Enter YouTube Video URL", "https://www.youtube.com/watch?v=your_video_id")

if youtube_url and st.button("Process Data"):
    st.write("Processing...")
    
    # Extract video ID and display the video only after processing
    video_id = youtube_url.split("v=")[-1].split("&")[0]

    # Columns layout for video and transcription
    col1, col2 = st.columns([1, 2])  # Make the video column smaller than transcription

    # Processing transcription and summarization
    transcription, summary = main(youtube_url)

    with col1:
        # Display YouTube video on the left side
        st.video(f"https://www.youtube.com/watch?v={video_id}")
    
    with col2:
        if transcription:
            st.subheader("Transcription")
            st.text_area("Transcription", transcription, height=300)

        if summary:
            st.subheader("Summary")
            st.text_area("Summary", summary, height=150)

    # Show video again at the bottom for emphasis after transcription and summary
    st.video(f"https://www.youtube.com/watch?v={video_id}")
