"""Best-effort Redis wake-ups; committed SQLite state is always authoritative."""
import asyncio
import logging

from redis.asyncio import Redis
from redis.exceptions import RedisError

LOG = logging.getLogger(__name__)


class RedisNotifications:
    def __init__(self, url):
        self.client = Redis.from_url(url, socket_connect_timeout=0.5, socket_timeout=0.5)

    async def publish(self):
        try:
            await self.client.publish("platform:changed", "1")
        except (RedisError, OSError):
            LOG.debug("Notification unavailable; database polling remains active")

    async def wait(self, timeout):
        try:
            async with asyncio.timeout(timeout):
                async with self.client.pubsub() as subscription:
                    await subscription.subscribe("platform:changed")
                    while True:
                        event = await subscription.get_message(ignore_subscribe_messages=True, timeout=timeout)
                        if event is not None:
                            return
        except TimeoutError:
            return
        except (RedisError, OSError):
            await asyncio.sleep(timeout)

    async def close(self):
        await self.client.aclose()
