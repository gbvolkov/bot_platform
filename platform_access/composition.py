from platform_application.service import Application
from platform_contracts.catalog import AgentCatalog
from platform_infrastructure.artifacts import ArtifactStorage
from platform_infrastructure.settings import PlatformSettings
from platform_infrastructure.sqlite import SQLiteRepository
from platform_infrastructure.notifications import RedisNotifications


def compose(settings=None, repository=None):
    settings = settings or PlatformSettings()
    repository = repository or SQLiteRepository(settings.database_path, lease_seconds=settings.lease_seconds)
    notifications = RedisNotifications(settings.redis_url) if settings.redis_url else None
    return (settings, Application(repository, settings.poll_seconds, notifications),
            AgentCatalog.load(settings.catalog_path), ArtifactStorage(settings.artifact_path, repository))
