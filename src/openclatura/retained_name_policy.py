"""Data-backed preferred output and accepted-alias policy for retained parents."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .naming_data import load_json_table


@dataclass(frozen=True)
class RetainedHydrogenationPolicy:
    base_parent: str
    hydro_locants: tuple[str, ...]
    indicated_hydrogen_locants: tuple[str, ...] = ()


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
    preferred_name = str(row["preferred_name"])
    if not template_name or not preferred_name:
        raise ValueError("retained parent name policy names must not be empty")
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
    return RetainedParentNamePolicy(
        template_name=template_name,
        preferred_name=preferred_name,
        accepted_aliases=tuple(str(alias) for alias in row.get("accepted_aliases", ())),
        preferred_contexts=tuple(str(context) for context in row.get("preferred_contexts", ("all",))),
        hydrogenation=hydrogenation,
        reason=str(row.get("reason", "")),
    )
