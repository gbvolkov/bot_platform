import os
import asyncio

from agents.tg_bots.bi_agent_bot import main


if __name__ == "__main__":
    pid = os.getpid()
    with open(".process", "w") as proc_file:
        proc_file.write(f"{pid}")
    asyncio.run(main())