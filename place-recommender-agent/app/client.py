import argparse
import asyncio
import os
import uuid
from pprint import pprint

from fasta2a.client import A2AClient, Message
from fasta2a.schema import MessageSendConfiguration, TextPart


DEFAULT_BASE_URL = "http://0.0.0.0:8000"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Utility client for the place recommender A2A app.")
    parser.add_argument(
        "--base-url",
        default=os.getenv("A2A_BASE_URL", DEFAULT_BASE_URL),
        help="A2A server base URL.",
    )
    parser.add_argument(
        "--message",
        help="User message to send to the agent.",
    )
    parser.add_argument(
        "--context-id",
        help="Optional context ID to continue an existing conversation.",
    )
    parser.add_argument(
        "--task-id",
        help="Task ID to fetch instead of sending a new message.",
    )
    return parser.parse_args()


def build_message(text: str, context_id: str | None) -> Message:
    payload = {
        "role": "user",
        "kind": "message",
        "message_id": str(uuid.uuid4()),
        "parts": [TextPart(kind="text", text=text)],
    }
    if context_id:
        payload["context_id"] = context_id
    return Message(**payload)


async def send_message(client: A2AClient, text: str, context_id: str | None) -> None:
    message = build_message(text=text, context_id=context_id)
    response = await client.send_message(
        message=message,
        configuration=MessageSendConfiguration(
            accepted_output_modes=["application/json", "text/plain"]
        ),
    )
    pprint(response)

    task_id = None
    if isinstance(response, dict):
        task_id = response.get("result", {}).get("id")

    if task_id:
        print(f"\nTask ID: {task_id}")


async def get_task(client: A2AClient, task_id: str) -> None:
    task = await client.get_task(task_id=task_id)
    pprint(task)


async def main() -> None:
    args = parse_args()
    client = A2AClient(base_url=args.base_url)

    if args.task_id:
        await get_task(client=client, task_id=args.task_id)
        return

    if not args.message:
        raise SystemExit("Provide either --message to send or --task-id to fetch a task.")

    await send_message(client=client, text=args.message, context_id=args.context_id)


if __name__ == "__main__":
    asyncio.run(main())
