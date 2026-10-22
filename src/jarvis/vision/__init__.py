"""Screen understanding: screenshot to vision model to answer.

``screen`` holds the ``ScreenReader`` service and the types tools use,
``imaging`` the downscale and byte-budget logic, ``factory`` the wiring from
``Settings``. Nothing is re-exported here so that importing the interface
types (as ``jarvis.agent.tool`` does) never pulls in the LLM providers.
"""
