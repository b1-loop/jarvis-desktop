"""Tool plugins, one file per capability, discovered automatically.

Dropping a module in this package that defines a ``Tool`` subclass adds
a capability; there is no central registration list. Tools must never
import each other or any concrete provider, only interfaces.
"""
