import asyncio
import time

import httpx

from platform_client.queue_adapter import DurableQueueClient


def test_quiet_run_keeps_connection_alive_without_restarting_read():
    async def check():
        release = asyncio.Event()

        class Stream(httpx.AsyncByteStream):
            reads = 0
            closed = False

            async def __aiter__(self):
                self.reads += 1
                await release.wait()
                yield b'data: {"type":"completed","payload":{"result":{"agent_message":{"raw_text":"done"}}}}\n\n'

            async def aclose(self):
                self.closed = True

        stream = Stream()
        client = DurableQueueClient('http://api/api', heartbeat_seconds=0.01,
            transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=stream)))
        client.jobs['job'] = ('run', 'user', time.monotonic())
        events = client.iter_events('job')
        assert (await asyncio.wait_for(anext(events), 1)).type == 'heartbeat'
        assert (await asyncio.wait_for(anext(events), 1)).type == 'heartbeat'
        assert stream.reads == 1
        release.set()
        terminal = await asyncio.wait_for(anext(events), 1)
        assert terminal.type == 'completed'
        assert terminal.metadata['content'] == 'done'
        await events.aclose()
        assert stream.closed
        await client.shutdown()

    asyncio.run(check())


def test_detaching_quiet_stream_closes_pending_read():
    async def check():
        stopped = asyncio.Event()

        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self):
                try:
                    await asyncio.Event().wait()
                    yield b''
                finally:
                    stopped.set()

        client = DurableQueueClient('http://api/api', heartbeat_seconds=0.01,
            transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=Stream())))
        client.jobs['job'] = ('run', 'user', time.monotonic())
        events = client.iter_events('job')
        assert (await asyncio.wait_for(anext(events), 1)).type == 'heartbeat'
        await events.aclose()
        assert stopped.is_set()
        await client.shutdown()

    asyncio.run(check())
