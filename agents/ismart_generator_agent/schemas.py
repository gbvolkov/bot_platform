from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class GeneratedMaterial(BaseModel):
    """Structured output returned by content subagents."""

    content: str = Field(
        default="",
        description="Final user-facing HTML. It must start with <style> and contain div.cc-lesson.",
    )
    agent_notes: list[str] = Field(
        default_factory=list,
        description="Short operational notes about which inputs were used.",
    )


class BlockIssue(BaseModel):
    block_id: str = Field(default="", description="Nearest heading id, anchor, or heading text.")
    block_heading: str = Field(default="", description="Human-readable block heading.")
    severity: Literal["blocking", "non_blocking"] = "blocking"
    issue: str = Field(default="", description="What is wrong in this block.")
    fix_instruction: str = Field(default="", description="Localized instruction for fixing this block.")


class PassedBlock(BaseModel):
    block_id: str = Field(default="", description="Nearest heading id, anchor, or heading text.")
    block_heading: str = Field(default="", description="Human-readable block heading.")
    reason: str = Field(default="", description="Why this block should be preserved on retry.")


class MaterialValidationDecision(BaseModel):
    """Structured output returned by MaterialValidatorAgent."""

    approved: bool = False
    issues: list[str] = Field(default_factory=list)
    fix_instructions: list[str] = Field(default_factory=list)
    issues_by_block: list[BlockIssue] = Field(default_factory=list)
    passed_blocks: list[PassedBlock] = Field(default_factory=list)


class ValidationControllerDecision(BaseModel):
    """Structured output returned by ValidationControllerAgent."""

    approved: bool = False
    decision: str = Field(default="keep_failed")
    quality_score: float = Field(default=0.0, ge=0.0, le=5.0)
    score_rationale: str = ""
    rationale: str = ""
    blocking_issues: list[str] = Field(default_factory=list)
    non_blocking_issues: list[str] = Field(default_factory=list)
    overruled_validator_issues: list[str] = Field(default_factory=list)
    residual_risks: list[str] = Field(default_factory=list)
    fix_instructions: list[str] = Field(default_factory=list)


class PackageValidationDecision(BaseModel):
    """Structured output returned by PackageValidatorAgent."""

    approved: bool = False
    issues: list[str] = Field(default_factory=list)
    fix_instructions: list[str] = Field(default_factory=list)
