"""Central assistant state machine: IDLE, LISTENING, THINKING, SPEAKING.

Every transition is validated and published on the event bus so the
overlay, logs and tray icon can never disagree about what Jarvis is
doing. Arrives in step 6.
"""
