"""Everything the assistant can do. Importing a module registers its tools."""

from sohayok.tools.registry import REGISTRY, Registry


def load_all() -> Registry:
    from sohayok.tools import (  # noqa: F401
        apps,
        browser,
        computer,
        devices,
        display,
        files,
        input,
        media,
        memory,
        notifications,
        screen,
        shell,
        system,
        timers,
        whatsapp,
        windows,
    )

    return REGISTRY
