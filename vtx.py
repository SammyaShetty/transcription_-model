import os
import wave
import math
import numpy as np
import speech_recognition as sr
from moviepy.editor import VideoFileClip

# Function to convert video to audio
def convert_video_to_audio(video_path, audio_path):
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path, codec='pcm_s16le')

# Function to enhance audio quality by increasing volume
def enhance_audio(input_audio_file, output_audio_file, volume_gain=1.5):
    with wave.open(input_audio_file, 'rb') as audio:
        params = audio.getparams()
        num_frames = audio.getnframes()
        
        with wave.open(output_audio_file, 'wb') as output_audio:
            output_audio.setparams(params)
            frames_per_read = 1024  # Number of frames to read at a time
            for _ in range(0, num_frames, frames_per_read):
                frames = audio.readframes(frames_per_read)
                enhanced_frames = enhance_audio_frames(frames, params.sampwidth, volume_gain)
                output_audio.writeframes(enhanced_frames)
    
    print(f"Enhanced audio temporarily created as {output_audio_file}")

def enhance_audio_frames(frames, sampwidth, volume_gain):
    if sampwidth == 1:  # 8-bit audio
        audio_samples = np.frombuffer(frames, dtype=np.uint8) - 128  # Convert to signed
    elif sampwidth == 2:  # 16-bit audio
        audio_samples = np.frombuffer(frames, dtype=np.int16)
    else:
        raise ValueError("Unsupported sample width")

    enhanced_samples = (audio_samples * volume_gain).astype(audio_samples.dtype)

    if sampwidth == 1:
        enhanced_samples = np.clip(enhanced_samples + 128, 0, 255)  # Convert back to unsigned
    elif sampwidth == 2:
        enhanced_samples = np.clip(enhanced_samples, -32768, 32767)

    return enhanced_samples.tobytes()

# Function to convert audio to text
def process_audio_chunk(chunk_filename, recognizer):
    with sr.AudioFile(chunk_filename) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            print(f"Transcribed text: {text}")  # Debug print
        except sr.UnknownValueError:
            text = " "
            print("Could not   derstand audio")
        except sr.RequestError as e:
            text = "[Error retrieving results]"
            print(f"Could not request results from Google Speech Recognition service; {e}")
        return text

def audio_to_text(input_audio_file, output_text_file, chunk_duration=60):
    recognizer = sr.Recognizer()
    with wave.open(input_audio_file, 'rb') as audio:
        params = audio.getparams()
        frame_rate = params.framerate
        n_channels = params.nchannels
        sampwidth = params.sampwidth
        n_frames = audio.getnframes()

        chunk_size = int(frame_rate * chunk_duration)
        total_chunks = math.ceil(n_frames / chunk_size)

        with open(output_text_file, 'w') as output_file:
            for i in range(total_chunks):
                chunk_filename = f"chunk_{i}.wav"
                with wave.open(chunk_filename, 'wb') as chunk:
                    chunk.setnchannels(n_channels)
                    chunk.setsampwidth(sampwidth)
                    chunk.setframerate(frame_rate)
                    frames = audio.readframes(chunk_size)
                    chunk.writeframes(frames)

                text = process_audio_chunk(chunk_filename, recognizer)
                output_file.write(text + "\n")
                print(f"Writing text to file: {text}")  # Debug print

                os.remove(chunk_filename)

    print(f"Audio conversion complete. Text saved to {output_text_file}")

# Main function to handle the entire process
def process_videos_in_folder(video_folder):
    for filename in os.listdir(video_folder):
        if filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            video_path = os.path.join(video_folder, filename)
            base_name = os.path.splitext(filename)[0]

            # Create paths for the audio and text files in the same folder
            audio_path = f"{base_name}_temp.wav"
            enhanced_audio_path = f"{base_name}_enhanced_temp.wav"
            output_text_file = os.path.join(video_folder, f"{base_name}.txt")

            # Convert video to audio
            convert_video_to_audio(video_path, audio_path)

            # Enhance audio quality
            enhance_audio(audio_path, enhanced_audio_path)

            # Convert enhanced audio to text
            audio_to_text(enhanced_audio_path, output_text_file)

            # Clean up: delete the temporary audio files
            os.remove(audio_path)
            os.remove(enhanced_audio_path)

            print(f"Processed video {filename} and saved output as {output_text_file}")

# Example usage
video_folder = "C:/Users/hp/Desktop/transcription/audio"
process_videos_in_folder(video_folder)
