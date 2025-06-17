import os
import asyncio
from dotenv import load_dotenv

from azure.identity.aio import DefaultAzureCredential

from semantic_kernel.agents import AzureAIAgent, ConcurrentOrchestration
from semantic_kernel.agents.runtime import InProcessRuntime

load_dotenv()

REVIEWER_NAME = "Reviewer"
WRITER_NAME = "Writer"

async def main():
    async with (DefaultAzureCredential() as creds, AzureAIAgent.create_client(credential=creds) as client,):

        # 1. Load agents from Azure AI Foundry
        writer_agent_definition = await client.agents.get_agent(
            agent_id="asst_rKiGOTSgGKgB14OeMtnVMoKT",
        )

        agent_writer = AzureAIAgent(
            client=client,
            definition=writer_agent_definition,
        )

        reviewer_agent_definition = await client.agents.get_agent(
            agent_id="asst_JFrVD7p2NZM3dDcWEnOn7nX3",
        )

        agent_reviewer = AzureAIAgent(
            client=client,
            definition=reviewer_agent_definition,
        )

        # 2. Create Concurrent Orchestration
        orchestration = ConcurrentOrchestration(members=[agent_writer, agent_reviewer])

        # 3. Start a runtime for orchestration
        runtime = InProcessRuntime()
        runtime.start()

        # 4. Define the shared task for all agents
        prompt = "Write a poem about the ocean."

        # 5. Run concurrent orchestration
        orchestration_result = await orchestration.invoke(
            task=prompt,
            runtime=runtime,
        )

        # 6. Gather and print results
        results = await orchestration_result.get(timeout=30)

        for result in results:
            print(f"# {result.role or 'Agent'} - {result.name or '*'}:\n{result.content}\n")

        # 7. Clean up runtime
        await runtime.stop_when_idle()

if __name__ == "__main__":
    asyncio.run(main())
