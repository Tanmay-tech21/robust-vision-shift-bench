"""Generate the reproducible weekly benchmark card and JSON snapshot."""

from __future__ import annotations

from vision_shift_bench import write_benchmark_artifacts


def main() -> None:
    json_path, markdown_path = write_benchmark_artifacts("artifacts")
    print(f"Wrote {markdown_path} and {json_path}")


if __name__ == "__main__":
    main()
