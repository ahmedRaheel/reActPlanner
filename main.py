"""Run indexing, one question, or an interactive question loop."""
import argparse
import asyncio
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from core import Model
from grounded_rag import Index, build_index, run

ROOT = Path(__file__).resolve().parent


async def main():
    load_dotenv(ROOT / ".env")
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(description="Real local grounded RAG")
    commands = parser.add_subparsers(dest="command")
    indexing = commands.add_parser("index", help="Index your real TXT/Markdown documents")
    indexing.add_argument("--rebuild", action="store_true")
    asking = commands.add_parser("ask", help="Ask one question")
    asking.add_argument("question")
    asking.add_argument("--web", action="store_true", help="Allow sending the question to Tavily as fallback")
    chatting = commands.add_parser("chat", help="Interactive independent questions")
    chatting.add_argument("--web", action="store_true")
    args = parser.parse_args()
    index = Index(Path(os.getenv("RAG_DB", str(ROOT / "data" / "rag.db"))))
    async with Model() as model:
        if args.command == "index":
            folder = Path(str(ROOT / "knowledge"))
            print(json.dumps(await build_index(model, folder, index, args.rebuild), indent=2))
        elif args.command == "ask":
            print(json.dumps(await run(model, args.question, allow_web=args.web, index=index),
                             indent=2, ensure_ascii=False))
        else:
            print("Ask about your indexed documents. Type exit to stop.")
            while True:
                question = await asyncio.to_thread(input, "\nYou: ")
                if question.strip().lower() in {"exit", "quit"}:
                    break
                if not question.strip():
                    continue
                try:
                    response = await run(model, question, allow_web=getattr(args, "web", False), index=index)
                    print(json.dumps(response, indent=2, ensure_ascii=False))
                except (RuntimeError, ValueError) as error:
                    print(f"Error: {error}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (RuntimeError, ValueError, OSError) as error:
        raise SystemExit(str(error))
    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye.")
