# Task List: AI Video Generator Improvements

## Critical Issues

- [ ] **Security: Remove Hardcoded API Keys**
  - **File:** `aiatt/full_code.py`
  - **Issue:** The Pexels API key is hardcoded in the `__main__` block.
  - **Fix:** Use `os.getenv('PEXELS_API_KEY')` and load from a `.env` file.

- [ ] **Path Independence**
  - **File:** `aiatt/full_code.py`
  - **Issue:** Output path is hardcoded to a specific user's Desktop (`C:/Users/bozku/...`).
  - **Fix:** Use relative paths or `os.path.expanduser("~")` dynamically, or better yet, output to a `output/` folder within the project.

## Improvements

- [ ] **Error Handling for Audio**
  - The script assumes specific audio files exist (`ozanabises2.mp3`). Add a check at the start to verify input files exist before proceeding.

- [ ] **Modularization**
  - The `full_code.py` is quite large. Consider splitting the Pexels downloading logic and the MoviePy editing logic into separate classes/files (`downloader.py`, `editor.py`).
