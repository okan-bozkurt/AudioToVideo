import whisper
import os
import requests
import random
import time
import glob
import math
import logging
from textblob import TextBlob
from moviepy.editor import VideoFileClip, AudioFileClip, TextClip, CompositeVideoClip, concatenate_videoclips
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def transcribe_audio(audio_path, max_duration=None):
    try:
        logger.info(f"Transcribing audio: {audio_path}")
        model = whisper.load_model("base")
        result = model.transcribe(audio_path, fp16=False)
        transcription = result["text"]
        with open("transcription.txt", "w", encoding="utf-8") as f:
            f.write(transcription)
        subtitles = [(seg["start"], seg["end"], seg["text"]) for seg in result["segments"]]
        with open("subtitles.srt", "w", encoding="utf-8") as f:
            for i, (start, end, text) in enumerate(subtitles, 1):
                start_h, start_m, start_s = int(start // 3600), int((start % 3600) // 60), start % 60
                end_h, end_m, end_s = int(end // 3600), int((end % 3600) // 60), end % 60
                f.write(
                    f"{i}\n{start_h:02d}:{start_m:02d}:{start_s:06.3f} --> {end_h:02d}:{end_m:02d}:{end_s:06.3f}\n{text}\n\n")
        blob = TextBlob(transcription)
        mood = "meditative" if blob.sentiment.polarity <= 0.1 else "motivational"
        logger.info(f"Transcription complete. Mood: {mood}")
        return transcription, subtitles, mood
    except Exception as e:
        logger.error(f"Error in transcribe_audio: {e}", exc_info=True)
        raise


def extract_quotes(segments, max_quotes=3):
    try:
        logger.info("Extracting quotes")
        key_phrases = [seg["text"] for seg in segments if len(seg["text"]) > 20]
        quotes = key_phrases[:max_quotes]
        logger.info(f"Extracted quotes: {quotes}")
        return quotes
    except Exception as e:
        logger.error(f"Error in extract_quotes: {e}", exc_info=True)
        raise


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((requests.Timeout, requests.ConnectionError)),
    reraise=True
)
def fetch_pexels_videos(url, headers):
    """Fetch videos from Pexels with retry logic."""
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    return response.json().get("videos", [])


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((requests.Timeout, requests.ConnectionError)),
    reraise=True
)
def download_pexels_video(video_url, headers, file_name):
    """Download a video from Pexels with retry logic."""
    video_response = requests.get(video_url, headers=headers, stream=True, timeout=10)
    video_response.raise_for_status()
    with open(file_name, "wb") as f:
        for chunk in video_response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return file_name


def fetch_stock_videos(mood, audio_duration, api_key, min_duration=12, max_duration=18):
    query = "nature meditative portrait" if mood == "meditative" else "success motivational portrait"
    per_page = 20
    max_pages = 20
    page = random.randint(1, max_pages)
    video_files = []
    total_duration = 0
    attempts = 0
    max_attempts = 10

    # Clear old videos
    for i in range(20):
        file_name = f"video_new_{i}_*.mp4"
        for file in glob.glob(file_name):
            try:
                os.remove(file)
                logger.info(f"Removed old video: {file}")
            except OSError as e:
                logger.warning(f"Error removing {file}: {e}")

    while total_duration < audio_duration and attempts < max_attempts:
        url = f"https://api.pexels.com/videos/search?query={query}&per_page={per_page}&page={page}&orientation=portrait"
        headers = {"Authorization": api_key}
        try:
            logger.info(f"Fetching videos for query: {query}, page: {page}, attempt: {attempts + 1}")
            videos = fetch_pexels_videos(url, headers)
            if not videos:
                logger.info("No videos found on this page")
                page = (page % max_pages) + 1
                attempts += 1
                continue
        except requests.RequestException as e:
            logger.error(f"Error fetching videos: {e}")
            page = (page % max_pages) + 1
            attempts += 1
            continue

        random.shuffle(videos)
        for i, video in enumerate(videos):
            video_duration = video.get("duration", 0)
            if not (min_duration <= video_duration <= max_duration):
                logger.debug(
                    f"Skipping video {video['id']}: duration {video_duration}s outside {min_duration}-{max_duration}s")
                continue
            if not video.get("video_files"):
                logger.debug(f"Skipping video {video['id']}: no video files available")
                continue
            video_file = next((vf for vf in video["video_files"] if vf["quality"] in ["hd", "full_hd"]),
                              video["video_files"][0])
            video_url = video_file["link"]
            file_name = f"video_new_{len(video_files)}_{int(time.time())}.mp4"
            try:
                logger.info(f"Downloading video {len(video_files) + 1}: {video_url}, duration: {video_duration}s")
                download_pexels_video(video_url, headers, file_name)
                if os.path.exists(file_name) and os.path.getsize(file_name) > 0:
                    video_files.append(file_name)
                    total_duration += video_duration
                    logger.info(f"Saved video: {file_name}, duration: {video_duration}s")
                    if total_duration >= audio_duration + min_duration:
                        break
                else:
                    logger.error(f"Failed to save video {file_name}: Empty or missing file")
                    os.remove(file_name) if os.path.exists(file_name) else None
            except requests.RequestException as e:
                logger.error(f"Error downloading video {i}: {e}")
                continue

        page = (page % max_pages) + 1
        attempts += 1

    if total_duration < audio_duration:
        logger.info(f"Additional videos needed: Current total duration ({total_duration}s) < audio ({audio_duration}s)")
        remaining_duration = audio_duration - total_duration
        additional_attempts = 0
        max_additional_attempts = 5
        while total_duration < audio_duration and additional_attempts < max_additional_attempts:
            page = random.randint(1, max_pages)
            url = f"https://api.pexels.com/videos/search?query={query}&per_page={per_page}&page={page}&orientation=portrait"
            headers = {"Authorization": api_key}
            try:
                logger.info(f"Fetching additional videos, page: {page}, attempt: {additional_attempts + 1}")
                videos = fetch_pexels_videos(url, headers)
                if not videos:
                    logger.info("No videos found on this page")
                    additional_attempts += 1
                    continue
            except requests.RequestException as e:
                logger.error(f"Error fetching additional videos: {e}")
                additional_attempts += 1
                continue

            random.shuffle(videos)
            for i, video in enumerate(videos):
                video_duration = video.get("duration", 0)
                if not (min_duration <= video_duration <= max_duration):
                    logger.debug(
                        f"Skipping video {video['id']}: duration {video_duration}s outside {min_duration}-{max_duration}s")
                    continue
                if not video.get("video_files"):
                    logger.debug(f"Skipping video {video['id']}: no video files available")
                    continue
                video_file = next((vf for vf in video["video_files"] if vf["quality"] in ["hd", "full_hd"]),
                                  video["video_files"][0])
                video_url = video_file["link"]
                file_name = f"video_new_{len(video_files)}_{int(time.time())}.mp4"
                try:
                    logger.info(
                        f"Downloading additional video {len(video_files) + 1}: {video_url}, duration: {video_duration}s")
                    download_pexels_video(video_url, headers, file_name)
                    if os.path.exists(file_name) and os.path.getsize(file_name) > 0:
                        video_files.append(file_name)
                        total_duration += video_duration
                        logger.info(f"Saved additional video: {file_name}, duration: {video_duration}s")
                        if total_duration >= audio_duration + min_duration:
                            break
                    else:
                        logger.error(f"Failed to save video {file_name}: Empty or missing file")
                        os.remove(file_name) if os.path.exists(file_name) else None
                except requests.RequestException as e:
                    logger.error(f"Error downloading additional video {i}: {e}")
                    continue

            additional_attempts += 1

    if total_duration < audio_duration:
        logger.error(
            f"Total video duration ({total_duration}s) is less than audio ({audio_duration}s) after all attempts")
        raise ValueError("Insufficient video duration to cover audio")

    logger.info(f"Fetched {len(video_files)} videos: {video_files}, total duration: {total_duration}s")
    return video_files


def verify_inputs(audio_path, video_files, audio_duration):
    try:
        logger.info(f"Verifying inputs: audio={audio_path}, videos={video_files}")
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file {audio_path} not found.")
        audio = AudioFileClip(audio_path)
        audio_duration_verified = audio.duration
        audio.close()
        if abs(audio_duration - audio_duration_verified) > 0.1:
            logger.warning(
                f"Provided audio duration ({audio_duration}s) differs from actual ({audio_duration_verified}s)")

        valid_video_files = []
        total_duration = 0
        logger.info(f"Found {len(video_files)} existing video files")
        for vf in video_files:
            if not os.path.exists(vf):
                logger.warning(f"Video file {vf} not found")
                continue
            try:
                clip = VideoFileClip(vf)
                if 12 <= clip.duration <= 18:
                    logger.info(f"Verified {vf}: {clip.w}x{clip.h}, duration: {clip.duration}s")
                    valid_video_files.append(vf)
                    total_duration += clip.duration
                else:
                    logger.info(f"Skipping {vf}: duration {clip.duration}s outside 12-18s")
                clip.close()
            except Exception as e:
                logger.error(f"Invalid video file {vf}: {e}")
                continue

        if total_duration < audio_duration:
            logger.warning(
                f"Total video duration ({total_duration}s) is less than audio ({audio_duration}s); will fetch new videos")

        logger.info(
            f"Input verification complete. Audio duration: {audio_duration}s, Valid videos: {len(valid_video_files)}, Total video duration: {total_duration}s")
        return audio_duration, valid_video_files, total_duration
    except Exception as e:
        logger.error(f"Error in verify_inputs: {e}", exc_info=True)
        raise


def create_video(audio_path, video_files, subtitles, quotes, output_path):
    try:
        logger.info(f"Creating video: {output_path}")
        output_dir = os.path.dirname(output_path)
        if not os.access(output_dir, os.W_OK):
            raise PermissionError(f"No write permission for {output_dir}")

        # Handle existing output file with retries
        final_output_path = output_path
        if os.path.exists(output_path):
            for attempt in range(3):
                try:
                    os.remove(output_path)
                    logger.info(f"Removed existing {output_path}")
                    break
                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1}: Cannot remove {output_path}: {e}")
                    if attempt == 2:
                        timestamp = int(time.time())
                        final_output_path = output_path.replace(".mp4", f"_{timestamp}.mp4")
                        logger.info(f"Using fallback output path: {final_output_path}")
                    time.sleep(1)

        audio = AudioFileClip(audio_path)
        logger.info(f"Using full audio duration: {audio.duration} seconds")
        clips = []
        total_video_duration = 0
        remaining_duration = audio.duration

        for i, vf in enumerate(video_files):
            try:
                clip = VideoFileClip(vf)
                logger.info(f"Processing {vf}: Original {clip.w}x{clip.h}, duration: {clip.duration}s")
                clip = clip.resize(height=1920)
                if clip.w > 1080:
                    clip = clip.crop(x_center=clip.w / 2, width=1080)
                if clip.w < 1080 or clip.h < 1920:
                    logger.warning(f"{vf} upscaled to 1080x1920, quality may be reduced")
                clip = clip.resize((1080, 1920))

                if total_video_duration + clip.duration > audio.duration:
                    clip = clip.subclip(0, remaining_duration)
                    logger.info(f"Trimmed {vf} to {clip.duration}s to match audio")

                total_video_duration += clip.duration
                remaining_duration -= clip.duration
                clips.append(clip)

                if total_video_duration >= audio.duration:
                    break
            except Exception as e:
                logger.error(f"Error processing {vf}: {e}")
                continue

        if total_video_duration < audio.duration:
            raise ValueError(f"Video duration ({total_video_duration}s) is less than audio ({audio.duration}s)")

        logger.info(f"Total video duration: {total_video_duration} seconds")
        video = concatenate_videoclips(clips, method="compose")
        if video.duration > audio.duration:
            video = video.subclip(0, audio.duration)

        if audio.duration > 90:
            logger.warning(f"Audio duration ({audio.duration}s) exceeds Instagram Reels' 90-second limit")

        def make_subtitle_clip(txt, start, end):
            try:
                txt = ''.join(c for c in txt if c.isprintable())
                return TextClip(txt, fontsize=24, color="white", bg_color="black", size=(1000, None),
                                method="caption").set_position(("center", "bottom")).set_start(start).set_end(end)
            except Exception as e:
                logger.error(f"Error creating subtitle clip for text '{txt}': {e}")
                raise

        subtitle_clips = []
        for start, end, text in subtitles:
            try:
                if start < audio.duration:
                    clip = make_subtitle_clip(text, start, min(end, audio.duration))
                    subtitle_clips.append(clip)
            except Exception as e:
                logger.error(f"Failed to create subtitle for text '{text}' at {start}-{end}: {e}")
                continue

        quote_clips = []
        for i, quote in enumerate(quotes):
            try:
                quote = ''.join(c for c in quote if c.isprintable())
                start_time = i * 15
                if start_time < audio.duration:
                    clip = TextClip(quote, fontsize=30, color="yellow", bg_color="black", size=(1000, None),
                                    method="caption").set_position(("center", 300)).set_duration(
                        min(5, audio.duration - start_time)).set_start(start_time)
                    quote_clips.append(clip)
            except Exception as e:
                logger.error(f"Failed to create quote clip for '{quote}': {e}")
                continue

        final_video = CompositeVideoClip([video] + subtitle_clips + quote_clips).set_audio(audio)
        final_video.write_videofile(
            final_output_path,
            fps=24,
            codec="libx264",
            audio_codec="aac",
            threads=2,  # Reduced threads for stability
            ffmpeg_params=["-strict", "-2"],
            logger="bar"
        )
        logger.info(f"Video created successfully at {final_output_path}")

        for clip in clips:
            if clip:
                clip.close()
        if video:
            video.close()
        if audio:
            audio.close()
        if final_video:
            final_video.close()
        return final_output_path
    except Exception as e:
        # Ensure resources are closed on error
        for clip in clips:
            if clip:
                clip.close()
        if 'video' in locals() and video:
            video.close()
        if 'audio' in locals() and audio:
            audio.close()
        if 'final_video' in locals() and final_video:
            final_video.close()
        logger.error(f"Error in create_video: {e}", exc_info=True)
        raise


def main(audio_path, output_path, video_files, pexels_api_key=None):
    try:
        # Step 1: Get audio duration
        audio = AudioFileClip(audio_path)
        audio_duration = audio.duration
        audio.close()
        logger.info(f"Audio duration: {audio_duration} seconds")

        # Step 2: Check existing videos
        existing_videos = glob.glob("video_new_*.mp4")
        audio_duration, valid_video_files, total_video_duration = verify_inputs(audio_path, existing_videos,
                                                                                audio_duration)

        # Fetch new videos if existing ones are insufficient
        if total_video_duration < audio_duration or not valid_video_files:
            logger.info("Existing videos insufficient or missing; fetching new videos")
            _, subtitles, mood = transcribe_audio(audio_path)
            valid_video_files = fetch_stock_videos(mood, audio_duration, pexels_api_key)
            audio_duration, valid_video_files, total_video_duration = verify_inputs(audio_path, valid_video_files,
                                                                                    audio_duration)
            if total_video_duration < audio_duration or not valid_video_files:
                raise ValueError("Failed to fetch sufficient videos to cover audio duration")
        else:
            logger.info(f"Using {len(valid_video_files)} existing videos, total duration: {total_video_duration}s")

        # Step 3: Transcribe and analyze
        transcription, subtitles, mood = transcribe_audio(audio_path)
        logger.info(f"Mood detected: {mood}")

        # Step 4: Extract quotes
        quotes = extract_quotes([{"text": s[2]} for s in subtitles])

        # Step 5: Create video
        output_file = create_video(audio_path, valid_video_files, subtitles, quotes, output_path)
        logger.info(f"Final video created at {os.path.abspath(output_file)}")
        return output_file
    except Exception as e:
        logger.error(f"Error in main: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    # Input/Output configuration
    audio_path = "ozanabises2.mp3"  # Replace with your audio file
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f"reels_video_{int(time.time())}.mp4")
    
    video_files = []
    
    # Load API Key securely
    pexels_api_key = os.getenv("PEXELS_API_KEY")
    
    if not pexels_api_key:
        logger.error("PEXELS_API_KEY not found in environment variables. Please create a .env file.")
    elif not os.path.exists(audio_path):
        logger.error(f"Audio file '{audio_path}' not found. Please place your mp3 file in the project directory.")
    else:
        main(audio_path, output_path, video_files, pexels_api_key)