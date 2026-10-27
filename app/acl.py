"""Document-level access control.

Documents get ``allowed_groups`` at ingestion time from an ACL file (``DATA/acl.yaml``)::

    default: [public]                 # groups for documents no rule matches
    rules:                            # first match wins; globs are relative to DATA/
      - match: "true_data/job_management.html"
        groups: [data-platform]
      - match: "hr/**"
        groups: [hr, managers]

Callers get groups from their API key (``API_KEYS="key1:hr|managers,key2"``; keys without groups
get ``ACL_DEFAULT_USER_GROUPS``). Retrieval only returns chunks whose ``allowed_groups`` intersect
the caller's groups. The group ``*`` means unrestricted (admin, evals).
"""

from __future__ import annotations

import fnmatch
import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

PUBLIC = "public"
ALL = "*"


@dataclass
class AclRule:
    match: str
    groups: list[str]


@dataclass
class AclPolicy:
    default: list[str] = field(default_factory=lambda: [PUBLIC])
    rules: list[AclRule] = field(default_factory=list)

    def groups_for(self, relative_path: str) -> list[str]:
        for rule in self.rules:
            if fnmatch.fnmatch(relative_path, rule.match):
                return sorted(set(rule.groups))
        return sorted(set(self.default))


def load_acl_policy(path: Path) -> AclPolicy:
    if not path.exists():
        return AclPolicy()
    raw = yaml.safe_load(path.read_text()) or {}
    policy = AclPolicy(
        default=[str(g) for g in raw.get("default", [PUBLIC])],
        rules=[AclRule(match=str(r["match"]), groups=[str(g) for g in r["groups"]]) for r in raw.get("rules", [])],
    )
    for group_list in [policy.default, *(r.groups for r in policy.rules)]:
        if not group_list:
            raise ValueError(f"{path}: every rule needs at least one group (an empty list would hide documents)")
    logger.info("Loaded ACL policy from %s (%d rules)", path, len(policy.rules))
    return policy


def is_unrestricted(groups: list[str] | None) -> bool:
    return groups is not None and ALL in groups


def effective_groups(groups: list[str] | tuple[str, ...]) -> list[str]:
    """Every caller can read public documents, in addition to their own groups."""
    return sorted({*groups, PUBLIC})
