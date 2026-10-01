"""Data-backed preferred output and accepted-alias policy for retained parents."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from functools import lru_cache

from .formatting import format_multiplier
from .locants import retained_locant_sort_key
from .name_operations import HydroOperation
from .naming_data import load_json_table

_INDICATED_H_PREFIX = re.compile(r"^(?:\d+[a-z]?H,)*\d+[a-z]?H-")


@dataclass(frozen=True)
class RetainedHydrogenationPolicy:
    base_parent: str
    hydro_locants: tuple[str, ...]
    indicated_hydrogen_locants: tuple[str, ...] = ()

    @property
    def operations(self) -> tuple[HydroOperation, ...]:
        operations = []
        if self.hydro_locants:
            operations.append(
                HydroOperation(
                    key="retained_parent_hydrogenation",
                    reason="Preferred retained parent expressed as an additive-hydrogen derivative.",
                    locants=self.hydro_locants,
                    operation_kind="additive_hydrogen",
                )
            )
        if self.indicated_hydrogen_locants:
            operations.append(
                HydroOperation(
                    key="retained_parent_indicated_hydrogen",
                    reason="Preferred retained parent requires cited indicated hydrogen.",
                    locants=self.indicated_hydrogen_locants,
                    operation_kind="indicated_hydrogen",
                )
            )
        return tuple(operations)

    def relocated(self, suffix_locants: frozenset[str]) -> RetainedHydrogenationPolicy:
        """Move the citation onto a suffix carbon the parent would hydrogenate.

        P-58.2.3.1: a suffix at a hydrogenated position takes the indicated
        hydrogen, and the positions it vacates become the hydro prefix. The
        hydrogenated set itself does not change -- only how it is split between
        the citation and the prefix -- so indoline's 2,3-dihydro-1H- becomes
        1,3-dihydro-2H- once the 2-one occupies C-2.

        A suffix on the cited position already agrees with the parent
        (2,3-dihydro-1H-inden-1-one), and a suffix off the hydrogenated set
        says nothing about it; both keep this policy unchanged.
        """

        moved = suffix_locants.intersection(self.hydro_locants)
        if not moved or len(moved) != len(self.indicated_hydrogen_locants):
            return self
        hydrogenated = set(self.hydro_locants) | set(self.indicated_hydrogen_locants)
        return replace(
            self,
            hydro_locants=tuple(sorted(hydrogenated - moved, key=retained_locant_sort_key)),
            indicated_hydrogen_locants=tuple(sorted(moved, key=retained_locant_sort_key)),
        )

    def render(self) -> str:
        # The base parent carries its own citation ("1H-indole"), which is the
        # one this policy may have moved, so drop it before citing the current
        # placement. Leaving it in spells the hydrogen twice: "1,3-dihydro-2H-1H-indole".
        parent = _INDICATED_H_PREFIX.sub("", self.base_parent, count=1)
        if self.indicated_hydrogen_locants:
            cited = ",".join(f"{locant}H" for locant in self.indicated_hydrogen_locants)
            parent = f"{cited}-{parent}"
        hydro = format_multiplier("hydro", len(self.hydro_locants))
        return f"{','.join(self.hydro_locants)}-{hydro}-{parent}"


@dataclass(frozen=True)
class RetainedParentNamePolicy:
    template_name: str
    preferred_name: str
    accepted_aliases: tuple[str, ...] = ()
    preferred_contexts: tuple[str, ...] = ("all",)
    hydrogenation: RetainedHydrogenationPolicy | None = None
    reason: str = ""
    context_names: tuple[tuple[str, str], ...] = ()

    @property
    def accepted_names(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((self.preferred_name, self.template_name, *self.accepted_aliases)))

    def output_name(self, context: str) -> str:
        contextual = dict(self.context_names).get(context)
        if contextual is not None:
            return contextual
        if "all" in self.preferred_contexts or context in self.preferred_contexts:
            return self.preferred_name
        return self.template_name


@lru_cache(maxsize=1)
def retained_parent_name_policies() -> tuple[RetainedParentNamePolicy, ...]:
    rows = load_json_table("retained_parent_name_policy.json").get("parents", ())
    policies = tuple(_policy_from_data(row) for row in rows)
    keys = [policy.template_name for policy in policies]
    if len(keys) != len(set(keys)):
        raise ValueError("retained parent name policies must have unique template_name values")
    return policies


@lru_cache(maxsize=1)
def _policies_by_template_name() -> dict[str, RetainedParentNamePolicy]:
    return {policy.template_name: policy for policy in retained_parent_name_policies()}


def retained_parent_name_policy(template_name: str) -> RetainedParentNamePolicy | None:
    return _policies_by_template_name().get(template_name)


def retained_parent_output_name(
    template_name: str,
    context: str,
    *,
    default_indicated_h: tuple[str, ...] = (),
    indicated_h: tuple[str, ...] | None = None,
) -> str:
    policy = retained_parent_name_policy(template_name)
    name = policy.output_name(context) if policy is not None else template_name
    return render_retained_hydrogen_state(name, default_indicated_h, indicated_h)


def render_retained_hydrogen_state(
    name: str, default_indicated_h: tuple[str, ...], indicated_h: tuple[str, ...] | None
) -> str:
    """Render a matched H state using declared citation metadata, not lexical inference."""
    if indicated_h is None or indicated_h == default_indicated_h:
        return name
    old_prefix = ",".join(f"{locant}H" for locant in default_indicated_h) + "-" if default_indicated_h else ""
    if old_prefix and name.startswith(old_prefix):
        name = name[len(old_prefix) :]
    prefix = ",".join(f"{locant}H" for locant in indicated_h) + "-" if indicated_h else ""
    return prefix + name


def _policy_from_data(row: dict) -> RetainedParentNamePolicy:
    template_name = str(row["template_name"])
    declared_preferred_name = str(row.get("preferred_name", ""))
    if not template_name:
        raise ValueError("retained parent template_name must not be empty")
    hydrogenation_data = row.get("hydrogenation")
    hydrogenation = None
    if hydrogenation_data is not None:
        hydrogenation = RetainedHydrogenationPolicy(
            base_parent=str(hydrogenation_data["base_parent"]),
            hydro_locants=tuple(str(locant) for locant in hydrogenation_data.get("hydro_locants", ())),
            indicated_hydrogen_locants=tuple(
                str(locant) for locant in hydrogenation_data.get("indicated_hydrogen_locants", ())
            ),
        )
        if not hydrogenation.hydro_locants:
            raise ValueError(f"hydrogenation policy for {template_name!r} has no hydro locants")
    preferred_name = hydrogenation.render() if hydrogenation is not None else declared_preferred_name
    if not preferred_name:
        raise ValueError("retained parent preferred_name must not be empty")
    if declared_preferred_name and declared_preferred_name != preferred_name:
        raise ValueError(
            f"retained parent policy for {template_name!r} declares {declared_preferred_name!r}, "
            f"but its operations render {preferred_name!r}"
        )
    return RetainedParentNamePolicy(
        template_name=template_name,
        preferred_name=preferred_name,
        accepted_aliases=tuple(str(alias) for alias in row.get("accepted_aliases", ())),
        preferred_contexts=tuple(str(context) for context in row.get("preferred_contexts", ("all",))),
        hydrogenation=hydrogenation,
        reason=str(row.get("reason", "")),
        context_names=tuple((str(context), str(name)) for context, name in row.get("context_names", {}).items()),
    )
