import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="video-ranker")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init-db", help="enable pgvector and create tables")
    ingest = sub.add_parser("ingest", help="download and register one or more videos")
    ingest.add_argument("urls", nargs="+")
    sub.add_parser("keyframes", help="extract keyframes for downloaded videos")
    sub.add_parser("transcribe", help="transcribe audio for downloaded videos")
    sub.add_parser("embed", help="embed keyframes and transcripts")
    sub.add_parser("caption", help="caption a few keyframes per video with a small VLM")

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
    elif args.command == "keyframes":
        from video_ranker.ingest.pipeline import keyframes_for_pending

        for video_id, count in keyframes_for_pending():
            print(f"video {video_id} -> {count} keyframes")
    elif args.command == "transcribe":
        from video_ranker.ingest.pipeline import transcribe_pending

        for video_id, count in transcribe_pending():
            print(f"video {video_id} -> {count} segments")
    elif args.command == "embed":
        from video_ranker.ingest.pipeline import embed_pending

        for video_id, count in embed_pending():
            print(f"video {video_id} -> {count} frames embedded")
    elif args.command == "caption":
        from video_ranker.ingest.pipeline import caption_pending

        for video_id, count in caption_pending():
            print(f"video {video_id} -> {count} captions")


if __name__ == "__main__":
    main()
