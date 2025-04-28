import os

import asyncio

from dotenv import load_dotenv

from semantic_kernel import Kernel
from semantic_kernel.agents import AgentGroupChat, ChatCompletionAgent
from semantic_kernel.agents.strategies import (
    KernelFunctionSelectionStrategy,
    KernelFunctionTerminationStrategy,
)
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents import ChatHistoryTruncationReducer
from semantic_kernel.functions import KernelFunctionFromPrompt

import asyncio

from azure.identity.aio import DefaultAzureCredential

from semantic_kernel.agents import AzureAIAgent, AzureAIAgentThread

load_dotenv()

REVIEWER_NAME = "Reviewer"
WRITER_NAME = "Writer"

async def main() -> None:
    async with (DefaultAzureCredential() as creds, AzureAIAgent.create_client(credential=creds) as client,):

        reviewer_agent_definition = await client.agents.get_agent(
            agent_id="agent_id",
        )

        writer_agent_definition = await client.agents.get_agent(
            agent_id="agent_id",
        )

        agent_reviewer = AzureAIAgent(
            client=client,
            definition=reviewer_agent_definition,
        )

        agent_writer = AzureAIAgent(
            client=client,
            definition=writer_agent_definition,
        )

        selection_function = KernelFunctionFromPrompt(
            function_name="selection",
            prompt=f"""
                Examine the provided RESPONSE and choose the next participant.
                State only the name of the chosen participant without explanation.
                Never choose the participant named in the RESPONSE.

                Choose only from these participants:
                - {REVIEWER_NAME}
                - {WRITER_NAME}

                Rules:
                - If RESPONSE is user input, it is {REVIEWER_NAME}'s turn.
                - If RESPONSE is by {REVIEWER_NAME}, it is {WRITER_NAME}'s turn.
                - If RESPONSE is by {WRITER_NAME}, it is {REVIEWER_NAME}'s turn.

                RESPONSE:
                {{{{$lastmessage}}}}
            """,
        )

        termination_keyword = "yes"

        termination_function = KernelFunctionFromPrompt(
            function_name="termination",
            prompt=f"""
                Examine the RESPONSE and determine whether the content has been deemed satisfactory.
                If the content is satisfactory, respond with a single word without explanation: {termination_keyword}.
                If specific suggestions are being provided, it is not satisfactory.
                If no correction is suggested, it is satisfactory.

                RESPONSE:
                {{{{$lastmessage}}}}
            """,
        )

        history_reducer = ChatHistoryTruncationReducer(target_count=5)

        kernel = Kernel()

        chat_service = AzureChatCompletion(
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            api_key=os.getenv("AZURE_AI_API_KEY"),
            endpoint=os.getenv("AZURE_AI_AGENT_ENDPOINT"),
        )

        kernel.add_service(chat_service)

        chat = AgentGroupChat(
            agents=[agent_reviewer, agent_writer],
            selection_strategy=KernelFunctionSelectionStrategy(
                initial_agent=agent_reviewer,
                function=selection_function,
                kernel=kernel,
                result_parser=lambda result: str(result.value[0]).strip() if result.value[0] is not None else WRITER_NAME,
                history_variable_name="lastmessage",
                history_reducer=history_reducer,
            ),
            termination_strategy=KernelFunctionTerminationStrategy(
                agents=[agent_reviewer],
                function=termination_function,
                kernel=kernel,
                result_parser=lambda result: termination_keyword in str(result.value[0]).lower(),
                history_variable_name="lastmessage",
                maximum_iterations=10,
                history_reducer=history_reducer,
            ),
        )

        try:

            await chat.add_chat_message(message="Write a poem about the ocean.")

            async for content in chat.invoke():
                print(f"# {content.role} - {content.name or '*'}: '{content.content}'")
        except Exception as e:
            print(f"An error occurred: {e}")

        await chat.reset()

if __name__ == "__main__":
    asyncio.run(main())
