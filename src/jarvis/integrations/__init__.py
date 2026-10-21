"""External service providers (email, calendar, music, chat, smart home).

Each service has an interface and one or more providers. Providers that
cannot be exercised without credentials ship as ``Stub*`` classes that
say so in their docstring and never pretend to succeed.
"""
