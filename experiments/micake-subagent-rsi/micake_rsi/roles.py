"""Specialist roles and their policy bindings (base model + role adapter + endpoint).

Every specialist call is bound to ``base_model_id / adapter_id / adapter_version
/ role / policy_version`` so trajectories can be attributed to the exact
adapter that produced them. Phase 1 serves each role adapter as an ATOM
merged checkpoint, so a binding also carries the endpoint serving it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from lumenrl.agentic.core.trajectory import PolicyTurn
from lumenrl.agentic.rollout import AgentPolicy, OpenAIChatPolicy

ROLES = ("planner", "critic")


@dataclass(frozen=True)
class Endpoint:
    """OpenAI-compatible endpoint serving one policy."""

    base_url: str
    model: str
    api_key_env: str | None
    verify_tls: bool
    timeout_s: float
    max_retries: int
    sampling: Mapping[str, Any]
    extra_body: Mapping[str, Any]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Endpoint":
        return cls(
            base_url=value["base_url"],
            model=value["model"],
            api_key_env=value.get("api_key_env"),
            verify_tls=bool(value["verify_tls"]),
            timeout_s=float(value["timeout_s"]),
            max_retries=int(value["max_retries"]),
            sampling=dict(value["sampling"]),
            extra_body=dict(value["extra_body"]),
        )


@dataclass(frozen=True)
class PolicyBinding:
    """Identity of the weights behind a policy."""

    role: str
    base_model_id: str
    adapter_id: str
    adapter_version: str
    policy_version: str
    trainable: bool

    @classmethod
    def from_dict(cls, role: str, value: Mapping[str, Any]) -> "PolicyBinding":
        return cls(
            role=role,
            base_model_id=value["base_model_id"],
            adapter_id=value["adapter_id"],
            adapter_version=str(value["adapter_version"]),
            policy_version=str(value["policy_version"]),
            trainable=bool(value["trainable"]),
        )


@dataclass(frozen=True)
class RoleSpec:
    """Configuration of one specialist role."""

    binding: PolicyBinding
    endpoint: Endpoint
    max_turns: int
    max_submissions: int

    @property
    def name(self) -> str:
        return self.binding.role


class BoundPolicy(AgentPolicy):
    """Wraps a policy so every recorded episode carries its weight binding."""

    def __init__(self, inner: AgentPolicy, binding: PolicyBinding) -> None:
        self.inner = inner
        self.binding = binding

    async def act(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        turn_index: int,
    ) -> PolicyTurn:
        return await self.inner.act(messages, tools, turn_index)

    def describe(self) -> dict[str, Any]:
        return {**self.inner.describe(), "binding": asdict(self.binding)}

    async def close(self) -> None:
        await self.inner.close()


def build_endpoint_policy(endpoint: Endpoint, binding: PolicyBinding) -> BoundPolicy:
    """OpenAI-compatible policy for ``endpoint`` tagged with ``binding``."""
    inner = OpenAIChatPolicy(
        base_url=endpoint.base_url,
        model=endpoint.model,
        api_key_env=endpoint.api_key_env,
        sampling=endpoint.sampling,
        timeout_s=endpoint.timeout_s,
        max_retries=endpoint.max_retries,
        verify_tls=endpoint.verify_tls,
        extra_body=endpoint.extra_body,
    )
    return BoundPolicy(inner, binding)


def load_role_specs(value: Mapping[str, Any]) -> dict[str, RoleSpec]:
    """Parse ``rsi.specialists`` into role specs; unknown roles are rejected."""
    specs: dict[str, RoleSpec] = {}
    for role, cfg in value.items():
        if role not in ROLES:
            raise ValueError(f"unknown specialist role {role!r}; supported: {ROLES}")
        specs[role] = RoleSpec(
            binding=PolicyBinding.from_dict(role, cfg["binding"]),
            endpoint=Endpoint.from_dict(cfg["endpoint"]),
            max_turns=int(cfg["max_turns"]),
            max_submissions=int(cfg["max_submissions"]),
        )
    return specs
