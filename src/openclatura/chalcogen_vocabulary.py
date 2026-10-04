"""Rule identifiers for graph-classified chalcogen functional groups.

This module is the narrow boundary between structural descriptors and the
existing functional-group rule registry.  Chemistry code selects with typed
roles; complete nomenclature keys are table values, never assembled by editing
another group's name.
"""

from .chalcogen_roles import Chalcogen, DerivativeKind, FunctionalFamily


_SIMPLE_KEYS: dict[tuple[FunctionalFamily, DerivativeKind, Chalcogen, bool], str] = {
    (FunctionalFamily.HYDROXY, DerivativeKind.ALCOHOL, Chalcogen.OXYGEN, False): "alcohol",
    (FunctionalFamily.HYDROXY, DerivativeKind.ALCOHOL, Chalcogen.SULFUR, False): "thiol",
    (FunctionalFamily.HYDROXY, DerivativeKind.ALCOHOL, Chalcogen.SELENIUM, False): "selenol",
    (FunctionalFamily.HYDROXY, DerivativeKind.ALCOHOL, Chalcogen.TELLURIUM, False): "tellurol",
    (FunctionalFamily.HYDROXY, DerivativeKind.ANION, Chalcogen.OXYGEN, False): "olate",
    (FunctionalFamily.HYDROXY, DerivativeKind.ANION, Chalcogen.SULFUR, False): "thiolate",
    (FunctionalFamily.HYDROXY, DerivativeKind.ANION, Chalcogen.SELENIUM, False): "selenolate",
    (FunctionalFamily.HYDROXY, DerivativeKind.ANION, Chalcogen.TELLURIUM, False): "tellurolate",
    (FunctionalFamily.CARBONYL, DerivativeKind.ALDEHYDE, Chalcogen.OXYGEN, False): "aldehyde",
    (FunctionalFamily.CARBONYL, DerivativeKind.ALDEHYDE, Chalcogen.SULFUR, False): "thioaldehyde",
    (FunctionalFamily.CARBONYL, DerivativeKind.ALDEHYDE, Chalcogen.SELENIUM, False): "selenoaldehyde",
    (FunctionalFamily.CARBONYL, DerivativeKind.ALDEHYDE, Chalcogen.TELLURIUM, False): "telluroaldehyde",
    (FunctionalFamily.CARBONYL, DerivativeKind.ALDEHYDE, Chalcogen.OXYGEN, True): "ring_aldehyde",
    (FunctionalFamily.CARBONYL, DerivativeKind.ALDEHYDE, Chalcogen.SULFUR, True): "ring_thioaldehyde",
    (FunctionalFamily.CARBONYL, DerivativeKind.ALDEHYDE, Chalcogen.SELENIUM, True): "ring_selenoaldehyde",
    (FunctionalFamily.CARBONYL, DerivativeKind.ALDEHYDE, Chalcogen.TELLURIUM, True): "ring_telluroaldehyde",
    (FunctionalFamily.CARBONYL, DerivativeKind.KETONE, Chalcogen.OXYGEN, False): "ketone",
    (FunctionalFamily.CARBONYL, DerivativeKind.KETONE, Chalcogen.SULFUR, False): "thioketone",
    (FunctionalFamily.CARBONYL, DerivativeKind.KETONE, Chalcogen.SELENIUM, False): "selenoketone",
    (FunctionalFamily.CARBONYL, DerivativeKind.KETONE, Chalcogen.TELLURIUM, False): "telluroketone",
    (FunctionalFamily.CHALCOGEN_ETHER, DerivativeKind.NEUTRAL_LINK, Chalcogen.OXYGEN, False): "ether",
    (FunctionalFamily.CHALCOGEN_ETHER, DerivativeKind.NEUTRAL_LINK, Chalcogen.SULFUR, False): "thioether",
    (FunctionalFamily.CHALCOGEN_ETHER, DerivativeKind.NEUTRAL_LINK, Chalcogen.SELENIUM, False): "selenoether",
    (FunctionalFamily.CHALCOGEN_ETHER, DerivativeKind.NEUTRAL_LINK, Chalcogen.TELLURIUM, False): "telluroether",
}


def simple_group_key(
    family: FunctionalFamily,
    derivative: DerivativeKind,
    element: Chalcogen,
    *,
    external: bool = False,
) -> str | None:
    """Resolve a validated simple-family descriptor to a registry key."""

    return _SIMPLE_KEYS.get((family, derivative, element, external))
