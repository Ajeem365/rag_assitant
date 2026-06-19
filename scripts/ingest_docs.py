import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table
from rich.progress import track

console = Console()


def main():
    parser = argparse.ArgumentParser(description="RAG Assistant — Document Ingestion CLI")
    parser.add_argument("--dir", default="docs", help="Directory of .md/.txt files")
    parser.add_argument("--url", action="append", default=[], help="URL(s) to ingest")
    parser.add_argument("--stats", action="store_true", help="Show corpus stats and exit")
    args = parser.parse_args()

    from app.db.chroma import collection_stats, get_vectorstore

    if args.stats:
        stats = collection_stats()
        console.print(f"\n[bold cyan]Corpus Stats[/bold cyan]")
        console.print(f"  Total chunks: [bold]{stats['total_chunks']}[/bold]")

        vs = get_vectorstore()
        result = vs.get(include=["metadatas"])
        source_counts: dict[str, int] = {}
        for meta in result.get("metadatas", []):
            src = meta.get("source", "unknown")
            source_counts[src] = source_counts.get(src, 0) + 1

        table = Table(title="Indexed Sources")
        table.add_column("Source", style="cyan")
        table.add_column("Chunks", justify="right", style="green")
        for src, count in sorted(source_counts.items()):
            table.add_row(src, str(count))
        console.print(table)
        return

    from app.ingestion.ingest import ingest_documents

    file_paths = []
    docs_dir = Path(args.dir)
    if docs_dir.exists():
        file_paths = [str(p) for p in docs_dir.glob("*.md")] + \
                     [str(p) for p in docs_dir.glob("*.txt")]

    if not file_paths and not args.url:
        console.print("[red]No documents found. Use --dir or --url.[/red]")
        sys.exit(1)

    console.print(f"\n[bold]Ingesting:[/bold]")
    for p in file_paths:
        console.print(f"  📄 {p}")
    for u in args.url:
        console.print(f"  🌐 {u}")

    result = ingest_documents(file_paths=file_paths, urls=args.url or None)

    console.print(f"\n[bold green]✅ Ingestion complete![/bold green]")
    console.print(f"  Documents loaded : {result['documents_loaded']}")
    console.print(f"  Chunks indexed   : {result['chunks_indexed']}")
    console.print(f"  Avg chunk size   : {result['avg_chunk_size']} chars")
    console.print(f"  Sources          : {', '.join(result['sources'])}")


if __name__ == "__main__":
    main()
