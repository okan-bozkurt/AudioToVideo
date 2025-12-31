# AI Automated Video Generator

This project is an automated pipeline that generates meditation and motivational videos by combining AI-generated audio/transcriptions with stock footage.

## Features

- **Audio Transcription:** Uses OpenAI's Whisper model to transcribe audio files and generate subtitles (`.srt`).
- **Mood Analysis:** Analyzes the sentiment of the transcription to determine the video mood (Meditative vs. Motivational).
- **Stock Footage Fetching:** Automatically searches and downloads relevant vertical videos from Pexels API based on the detected mood.
- **Video Assembly:** Uses `MoviePy` to stitch together video clips, overlay subtitles, and add quotes as text overlays.
- **Smart Duration Matching:** Ensures video clips loop or extend to match the exact length of the audio track.

## Directory Structure

- `aiatt/`: Contains the main Python source code.
  - `full_code.py`: The main orchestration script.
  - `video_generation.py`: Video processing logic.
  - `telegram_bot.py`: Telegram integration (if applicable).

## Requirements

- Python 3.8+
- [FFmpeg](https://ffmpeg.org/) (Required for MoviePy)

### Python Packages

```bash
pip install openai-whisper moviepy requests textblob tenacity
```

## Setup

1. **API Keys:**
   You need a Pexels API key. Create a `.env` file or export it as an environment variable (recommended) instead of hardcoding it.

2. **File Placement:**
   Place your input audio file in the project directory.

## Usage

Run the main script:

```bash
python aiatt/full_code.py
```

*Note: You may need to adjust the `audio_path` and `output_path` variables in the `__main__` block of `full_code.py` before running.*
