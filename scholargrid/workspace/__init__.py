"""Query-scoped topic workspace: one user query -> retrieved papers -> 4-12 sub-topics (+ micro-topics)
with keyword and plain-language labels, activity, top facets and a topic-specific reading path.
The global area map is too coarse for a beginner; they need the sub-structure of their own topic."""
from __future__ import annotations

from .build import build_workspace, load_cached, save_cached, workspace_key
from .model import Subtopic, TopicWorkspace, WsPaper

__all__ = ["TopicWorkspace", "Subtopic", "WsPaper", "build_workspace", "workspace_key", "load_cached",
           "save_cached"]
