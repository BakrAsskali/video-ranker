import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="video-ranker")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init-db", help="enable pgvector and create tables")
    ingest = sub.add_parser("ingest", help="download and register one or more videos")
    ingest.add_argument("urls", nargs="+")

    args = parser.parse_args()

    if args.command == "init-db":
        from video_ranker.db.session import init_db

        init_db()
        print("database ready")
    elif args.command == "ingest":
        from video_ranker.ingest.pipeline import ingest_url

        for url in args.urls:
            try:
                print(f"{url} -> video {ingest_url(url)}")
            except Exception as exc:
                print(f"{url} -> FAILED: {exc}")


if __name__ == "__main__":
    main()
