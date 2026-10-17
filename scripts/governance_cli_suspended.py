"""Display the suspension of a governance CLI without running its checks."""


def report_suspended(tool: str) -> int:
    print(f"{tool}: DISABLED; governance suspended; no checks executed")
    return 0
