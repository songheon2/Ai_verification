"""Bounded in-memory question history for one running analysis session."""
from collections import deque
import json

from Automation.LogAnalysisIndex import safe_data


class AnalysisConversation:
    def __init__(self):
        self.turns = deque(maxlen=6)

    def clear(self):
        self.turns.clear()

    def remember(self, question, answer):
        if not isinstance(answer, str) or not answer.strip():
            return
        if len(answer) > 3000:
            answer = answer[:2190] + "\n[…]\n" + answer[-800:]
        self.turns.append(safe_data({
            "question": (question.strip() or "기본 분석 요청")[:1000],
            "answer": answer,
        }))

    def snapshot(self):
        history = []
        for turn in reversed(self.turns):
            candidate = [turn, *history]
            if len(json.dumps(candidate, ensure_ascii=False).encode("utf-8")) > 20_000:
                break
            history = candidate
        return history
