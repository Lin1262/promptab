"""Statistically sound A/B testing for LLM prompts."""

from .stats import Comparison, Interval, compare, decide, plan, summarize

__all__ = ["Comparison", "Interval", "compare", "decide", "plan", "summarize"]
