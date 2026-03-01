from rich.console import Console
from rag import RAG

console = Console()

HELP = """
Commands:
  ask: <question>      -> ask a question over your documents
  help                -> show this help
  quit                -> exit

Setup:
  1) Put PDFs/TXT/MD into: data/docs
  2) Run: python ingest.py
  3) Run: python app.py
"""

def main():
    rag = RAG()
    console.print("[bold green]RAG Assistant ready.[/bold green]")
    console.print(HELP)

    while True:
        user = console.input("\n[bold cyan]You:[/bold cyan] ").strip()
        if not user:
            continue

        low = user.lower()

        if low == "quit":
            break
        if low == "help":
            console.print(HELP)
            continue

        if low.startswith("ask:"):
            q = user.split("ask:", 1)[1].strip()
            contexts = rag.retrieve(q, k=4)
            answer = rag.generate(q, contexts)   #  FIXED (single return value)
            console.print("\n[bold]Answer:[/bold]")
            console.print(answer)
            continue

        console.print("[yellow]Use: ask: <question>  or  help  or  quit[/yellow]")

if __name__ == "__main__":
    main()
