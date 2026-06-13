from __future__ import annotations

from app.container import AppContainer
from core.application.benchmarking.models import ModelProfile
from core.config.settings import Settings


def build_container_for_profile(
    profile: ModelProfile,
    base: Settings | None = None,
) -> AppContainer:
    settings = base or Settings()
    return AppContainer(
        settings=settings.model_copy(
            update={
                "llm_provider": profile.llm_synthesis_provider,
                "llm_routing_provider": profile.llm_routing_provider,
                "llm_synthesis_provider": profile.llm_synthesis_provider,
                "embedding_provider": profile.embedding_provider,
            }
        )
    )
