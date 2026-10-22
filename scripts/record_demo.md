# Recording the demo GIF

The README has a placeholder where `docs/media/demo.gif` belongs. It has to
be recorded by a person at the machine: the point of the GIF is a real
voice, a real answer and a real interruption, and nothing in this
repository fakes one. These steps take about ten minutes.

## 1. Prepare the machine

1. Ollama is running and has the model: `ollama pull qwen2.5:7b`.
2. Native build and models are in place:

   ```powershell
   uv run python scripts/build_native.py --skip-tests
   uv run python scripts/download_models.py
   ```

3. Use headphones (no echo cancellation yet, so loud speakers can make
   Jarvis interrupt itself), close notifications (Focus assist on) and set
   the display to 100 % or 125 % scaling so the overlay text stays legible
   in a 900 px wide GIF.
4. Install a screen recorder that writes GIFs or MP4s, e.g. ScreenToGif
   (`winget install NickeManarin.ScreenToGif`).

## 2. Lay out the screen

Start Jarvis in a terminal:

```powershell
uv run jarvis
```

Then arrange three things inside one 1280 x 720 region:

- the terminal (left half), which prints `you> ...` and `Jarvis> ...`;
- the dashboard (right half): tray icon menu -> "Show dashboard";
- the overlay, which sits in the bottom-right corner of the primary
  monitor; move the region so the overlay is inside it.

## 3. Record this script (about 40 seconds)

1. Say **"Hey Jarvis"**, wait for the beep, then **"What time is it?"**
   The overlay turns to Listening, then Thinking, then Speaking; the
   dashboard shows the `current_time` tool call and the stage timings.
2. Say **"Hey Jarvis, tell me a story about a lighthouse keeper."**
   About two sentences in, talk over it: **"Hold on, what is two plus two?"**
   Jarvis stops mid-word and answers the new question; this is the
   barge-in shot, so keep it in.
3. Press **Ctrl+Alt+J** and ask **"Open Notepad."** to show the hotkey and
   computer control.
4. Say **"Goodbye."**

## 4. Export and commit

- Trim to the spoken parts, 12-15 fps, width 900 px, under 8 MB.
- Save as `docs/media/demo.gif`.
- In `README.md`, replace the placeholder block under the title with:

  ```markdown
  ![Jarvis answering, being interrupted, and opening Notepad](docs/media/demo.gif)
  ```

- Commit on a `docs/...` branch: `docs: add demo GIF`.
