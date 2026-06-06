"""Pydantic schemas used by agents that produce structured output.

The framework's primary artifact is still prose: each agent's natural-language
reasoning is what users read in the saved markdown reports and what the
downstream agents read as context.  Structured output is layered onto the
three decision-making agents (Research Manager, Trader, Portfolio Manager)
so that:

- Their outputs follow consistent section headers across runs and providers
- Each provider's native structured-output mode is used (json_schema for
  OpenAI/xAI, response_schema for Gemini, tool-use for Anthropic)
- Schema field descriptions become the model's output instructions, freeing
  the prompt body to focus on context and the rating-scale guidance
- A render helper turns the parsed Pydantic instance back into the same
  markdown shape the rest of the system already consumes, so display,
  memory log, and saved reports keep working unchanged
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared rating types
# ---------------------------------------------------------------------------


class PortfolioRating(str, Enum):
    """5-tier rating used by the Research Manager and Portfolio Manager."""

    BUY = "Buy"
    OVERWEIGHT = "Overweight"
    HOLD = "Hold"
    UNDERWEIGHT = "Underweight"
    SELL = "Sell"


class TraderAction(str, Enum):
    """3-tier transaction direction used by the Trader.

    The Trader's job is to translate the Research Manager's investment plan
    into a concrete transaction proposal: should the desk execute a Buy, a
    Sell, or sit on Hold this round.  Position sizing and the nuanced
    Overweight / Underweight calls happen later at the Portfolio Manager.
    """

    BUY = "Buy"
    HOLD = "Hold"
    SELL = "Sell"


# ---------------------------------------------------------------------------
# Research Manager
# ---------------------------------------------------------------------------


class ResearchPlan(BaseModel):
    """Structured investment plan produced by the Research Manager.

    Hand-off to the Trader: the recommendation pins the directional view,
    the rationale captures which side of the bull/bear debate carried the
    argument, and the strategic actions translate that into concrete
    instructions the trader can execute against.
    """

    recommendation: PortfolioRating = Field(
        description=(
            "The investment recommendation. Exactly one of Buy / Overweight / "
            "Hold / Underweight / Sell. Reserve Hold for situations where the "
            "evidence on both sides is genuinely balanced; otherwise commit to "
            "the side with the stronger arguments."
        ),
    )
    rationale: str = Field(
        description=(
            "Conversational summary of the key points from both sides of the "
            "debate, ending with which arguments led to the recommendation. "
            "Speak naturally, as if to a teammate."
        ),
    )
    strategic_actions: str = Field(
        description=(
            "Concrete steps for the trader to implement the recommendation, "
            "including position sizing guidance consistent with the rating."
        ),
    )


def render_research_plan(plan: ResearchPlan) -> str:
    """Render a ResearchPlan to markdown for storage and the trader's prompt context."""
    return "\n".join([
        f"**Recommendation**: {plan.recommendation.value}",
        "",
        f"**Rationale**: {plan.rationale}",
        "",
        f"**Strategic Actions**: {plan.strategic_actions}",
    ])


# ---------------------------------------------------------------------------
# Trader
# ---------------------------------------------------------------------------


class TraderProposal(BaseModel):
    """Structured transaction proposal produced by the Trader.

    The trader reads the Research Manager's investment plan and the analyst
    reports, then turns them into a concrete transaction: what action to
    take, the reasoning that justifies it, and the practical levels for
    entry, stop-loss, and sizing.
    """

    action: TraderAction = Field(
        description="The transaction direction. Exactly one of Buy / Hold / Sell.",
    )
    reasoning: str = Field(
        description=(
            "The case for this action, anchored in the analysts' reports and "
            "the research plan. Start with the key risk before stating the opportunity. "
            "Two to four sentences."
        ),
    )
    probability_assessment: Optional[str] = Field(
        default=None,
        description=(
            "Probability-based assessment of the short-term setup, e.g. "
            "'60-70% upside continuation if volume holds; lower if support breaks'."
        ),
    )
    risk_first_summary: Optional[str] = Field(
        default=None,
        description="Key risk and why it matters before discussing opportunity.",
    )
    invalidation_condition: Optional[str] = Field(
        default=None,
        description="Concrete condition that invalidates the proposed trade or thesis.",
    )
    reverse_case: Optional[str] = Field(
        default=None,
        description="The strongest opposite view and what would make it correct.",
    )
    entry_price: Optional[float] = Field(
        default=None,
        description="Optional entry price target in the instrument's quote currency.",
    )
    stop_loss: Optional[float] = Field(
        default=None,
        description="Optional stop-loss price in the instrument's quote currency.",
    )
    position_sizing: Optional[str] = Field(
        default=None,
        description="Optional sizing guidance, e.g. '5% of portfolio'.",
    )
    stop_loss_plan: Optional[str] = Field(
        default=None,
        description="Concrete stop-loss or exit plan, including level and execution condition.",
    )
    validity_horizon: Optional[str] = Field(
        default=None,
        description="How long this signal remains valid, e.g. '1-5 trading days'.",
    )


def render_trader_proposal(proposal: TraderProposal) -> str:
    """Render a TraderProposal to markdown.

    The trailing ``FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**`` line is
    preserved for backward compatibility with the analyst stop-signal text
    and any external code that greps for it.
    """
    parts = [
        f"**Action**: {proposal.action.value}",
        "",
        f"**Reasoning**: {proposal.reasoning}",
    ]
    if proposal.risk_first_summary:
        parts.extend(["", f"**Risk First**: {proposal.risk_first_summary}"])
    if proposal.probability_assessment:
        parts.extend(["", f"**Probability Assessment**: {proposal.probability_assessment}"])
    if proposal.invalidation_condition:
        parts.extend(["", f"**Invalidation Condition**: {proposal.invalidation_condition}"])
    if proposal.reverse_case:
        parts.extend(["", f"**Reverse Case**: {proposal.reverse_case}"])
    if proposal.entry_price is not None:
        parts.extend(["", f"**Entry Price**: {proposal.entry_price}"])
    if proposal.stop_loss is not None:
        parts.extend(["", f"**Stop Loss**: {proposal.stop_loss}"])
    if proposal.position_sizing:
        parts.extend(["", f"**Position Sizing**: {proposal.position_sizing}"])
    if proposal.stop_loss_plan:
        parts.extend(["", f"**Stop Loss Plan**: {proposal.stop_loss_plan}"])
    if proposal.validity_horizon:
        parts.extend(["", f"**Validity Horizon**: {proposal.validity_horizon}"])
    parts.extend([
        "",
        f"FINAL TRANSACTION PROPOSAL: **{proposal.action.value.upper()}**",
    ])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Portfolio Manager
# ---------------------------------------------------------------------------


class PortfolioDecision(BaseModel):
    """Structured output produced by the Portfolio Manager.

    The model fills every field as part of its primary LLM call; no separate
    extraction pass is required. Field descriptions double as the model's
    output instructions, so the prompt body only needs to convey context and
    the rating-scale guidance.
    """

    rating: PortfolioRating = Field(
        description=(
            "The final position rating. Exactly one of Buy / Overweight / Hold / "
            "Underweight / Sell, picked based on the analysts' debate."
        ),
    )
    executive_summary: str = Field(
        description=(
            "A concise action plan covering entry strategy, position sizing, "
            "key risk levels, and time horizon. Two to four sentences."
        ),
    )
    investment_thesis: str = Field(
        description=(
            "Detailed reasoning anchored in specific evidence from the analysts' "
            "debate. If prior lessons are referenced in the prompt context, "
            "incorporate them; otherwise rely solely on the current analysis. "
            "Start with risk and invalidation conditions before stating opportunity."
        ),
    )
    probability_assessment: Optional[str] = Field(
        default=None,
        description="Probability-based assessment of the final decision, avoiding absolute certainty.",
    )
    risk_first_summary: Optional[str] = Field(
        default=None,
        description="The primary risk, invalidation risk, or constraint that controls the final call.",
    )
    invalidation_condition: Optional[str] = Field(
        default=None,
        description="Concrete event, level, or evidence that would make the decision wrong.",
    )
    reverse_case: Optional[str] = Field(
        default=None,
        description="The strongest opposite case and its trigger conditions.",
    )
    stop_loss_plan: Optional[str] = Field(
        default=None,
        description="Exit or stop-loss logic for the final decision, including specific trigger if available.",
    )
    price_target: Optional[float] = Field(
        default=None,
        description="Optional target price in the instrument's quote currency.",
    )
    time_horizon: Optional[str] = Field(
        default=None,
        description="Optional recommended holding period, e.g. '3-6 months'.",
    )
    validity_horizon: Optional[str] = Field(
        default=None,
        description="How long the decision remains valid before required reassessment.",
    )


def render_pm_decision(decision: PortfolioDecision) -> str:
    """Render a PortfolioDecision back to the markdown shape the rest of the system expects.

    Memory log, CLI display, and saved report files all read this markdown,
    so the rendered output preserves the exact section headers (``**Rating**``,
    ``**Executive Summary**``, ``**Investment Thesis**``) that downstream
    parsers and the report writers already handle.
    """
    parts = [
        f"**Rating**: {decision.rating.value}",
        "",
        f"**Executive Summary**: {decision.executive_summary}",
        "",
        f"**Investment Thesis**: {decision.investment_thesis}",
    ]
    if decision.risk_first_summary:
        parts.extend(["", f"**Risk First**: {decision.risk_first_summary}"])
    if decision.probability_assessment:
        parts.extend(["", f"**Probability Assessment**: {decision.probability_assessment}"])
    if decision.invalidation_condition:
        parts.extend(["", f"**Invalidation Condition**: {decision.invalidation_condition}"])
    if decision.reverse_case:
        parts.extend(["", f"**Reverse Case**: {decision.reverse_case}"])
    if decision.stop_loss_plan:
        parts.extend(["", f"**Stop Loss Plan**: {decision.stop_loss_plan}"])
    if decision.price_target is not None:
        parts.extend(["", f"**Price Target**: {decision.price_target}"])
    if decision.time_horizon:
        parts.extend(["", f"**Time Horizon**: {decision.time_horizon}"])
    if decision.validity_horizon:
        parts.extend(["", f"**Validity Horizon**: {decision.validity_horizon}"])
    return "\n".join(parts)
