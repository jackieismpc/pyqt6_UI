from .cli import main


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[calibration][ERROR] {exc}")
        raise SystemExit(2)
