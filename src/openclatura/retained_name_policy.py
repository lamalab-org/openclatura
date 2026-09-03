"""Data-backed preferred output and accepted-alias policy for retained parents."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .formatting import format_multiplier
from .name_operations import HydroOperation
from .naming_data import load_json_table


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

    def render(self) -> str:
        parent = self.base_parent
        if self.indicated_hydrogen_locants:
            cited = ",".join(f"{locant}H" for locant in self.indicated_hydrogen_locants)
            if not parent.startswith(f"{cited}-"):
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

    @property
    def accepted_names(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((self.preferred_name, self.template_name, *self.accepted_aliases)))

    def output_name(self, context: str) -> str:
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


def retained_parent_output_name(template_name: str, context: str) -> str:
    policy = retained_parent_name_policy(template_name)
    return policy.output_name(context) if policy is not None else template_name


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
    )
