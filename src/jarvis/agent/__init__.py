"""Agent loop, tool registry, plugin discovery and confirmation policy.

The agent turns a transcript into tool calls and a spoken summary. It
enforces confirmation for side-effecting tools centrally so no tool can
forget to ask.
"""
