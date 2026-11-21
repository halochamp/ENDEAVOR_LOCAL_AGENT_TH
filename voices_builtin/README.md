# voices_builtin

Reference voices that ship with the agent (one folder per voice: `ref_voice.wav` + `ref_voice.txt` + optional `voice.json`).

- `default/` is **เสียงมาตรฐาน (VOX)**, the voice the Text to Speech tab offers first.
  - `ref_voice.wav`: 6.4 s, mono, 16-bit, 24 kHz. `ref_voice.txt`: its exact words.
  - Provenance: synthetic (AI-generated) audio the project owner created and confirmed as synthetic; it is not a recording of a real, identifiable person.
  - To use your own voice as the default, put a `default/` folder (same two files) under `workspace/text_to_speech/voices/`; it takes the place of this one.
  - Any other voice you add goes in its own folder under `workspace/text_to_speech/voices/` (never here: that folder is for what the repository ships).
- Reading with OmniVoice (the engine of this tab) is CC-BY-NC: personal and non-commercial use only.
