"""Everything the assistant can do. Importing a module registers its tools."""

from sohayok.tools.registry import REGISTRY, Registry


def load_all() -> Registry:
    from sohayok.tools import (  # noqa: F401
        apps,
        browser,
        files,
        input,
        media,
        screen,
        shell,
        system,
        timers,
        windows,
    )

    return REGISTRY
