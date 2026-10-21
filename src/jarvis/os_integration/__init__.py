"""Python facade over the native OS layer (``jarvis_os``).

Defines the ``Platform`` interface that tools depend on, so a tool never
touches Win32 directly and a macOS or Linux backend can be dropped in
later without changing a single tool.
"""
