"""LLM clients behind a common ``LLMClient`` interface.

Both the Anthropic client and the local llama.cpp client yield the same
stream of text deltas and tool-call requests, so the agent loop is
unaware of which backend is answering.
"""
