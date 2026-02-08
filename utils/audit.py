from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class AuditLog:
    steps_ok: Dict[str, bool] = field(default_factory=dict)
    sources: List[str] = field(default_factory=list)
    insufficiencies: List[str] = field(default_factory=list)
    divergences: List[str] = field(default_factory=list)

    def ok(self, step: str):
        self.steps_ok[step] = True

    def fail(self, step: str):
        self.steps_ok[step] = False

    def add_source(self, s: str):
        if s not in self.sources:
            self.sources.append(s)

    def add_insufficient(self, s: str):
        self.insufficiencies.append(s)

    def add_divergence(self, s: str):
        self.divergences.append(s)

    def render(self) -> str:
        lines = ["*AUDITORIA*"]
        for k in sorted(self.steps_ok.keys()):
            v = self.steps_ok[k]
            lines.append(f"- {k}: {'OK' if v else 'FALHA'}")
        if self.sources:
            lines.append("\n*Fontes*")
            for s in self.sources:
                lines.append(f"- {s}")
        if self.insufficiencies:
            lines.append("\n*Dados insuficientes*")
            for s in self.insufficiencies[:12]:
                lines.append(f"- {s}")
            if len(self.insufficiencies) > 12:
                lines.append(f"- (+{len(self.insufficiencies)-12}…)")

        if self.divergences:
            lines.append("\n*Divergências*")
            for s in self.divergences[:8]:
                lines.append(f"- {s}")
            if len(self.divergences) > 8:
                lines.append(f"- (+{len(self.divergences)-8}…)")

        return "\n".join(lines)
