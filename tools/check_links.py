"""Check relative Markdown links and anchors in the given files."""
import re
import sys
from pathlib import Path


def slug(heading: str) -> str:
    text = heading.strip().lower()
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def anchors(path: Path) -> set[str]:
    found = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^#{1,6}\s+(.*)$", line)
        if match:
            found.add(slug(match.group(1)))
    return found


bad = 0
for name in sys.argv[1:]:
    source = Path(name)
    text = source.read_text(encoding="utf-8")
    for target in re.findall(r"\]\(([^)\s]+)\)", text):
        if target.startswith(("http://", "https://", "#terminal")):
            continue
        file_part, _, anchor = target.partition("#")
        path = (source.parent / file_part).resolve() if file_part else source.resolve()
        if not path.exists():
            print(f"{name}: missing file {target}")
            bad += 1
        elif anchor and path.suffix == ".md" and anchor not in anchors(path):
            print(f"{name}: missing anchor {target}")
            bad += 1
print("bad links:", bad)
