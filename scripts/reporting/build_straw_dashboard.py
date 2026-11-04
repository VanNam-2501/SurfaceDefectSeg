"""Build a self-contained offline research dashboard with links to original files."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from apps.stream_demo.research_catalog import ResearchCatalog


def main():
    destination = ROOT / "artifacts/research_dashboard"
    destination.mkdir(parents=True, exist_ok=True)
    catalog = ResearchCatalog(ROOT).snapshot(static=True)
    asset = ROOT / "apps/stream_demo"
    css = (asset / "workspace.css").read_text(encoding="utf-8")
    js = (asset / "research_report.js").read_text(encoding="utf-8")
    data = json.dumps(catalog, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    page = (asset / "research_report.html").read_text(encoding="utf-8")
    page = page.replace('<script src="/workspace_diagnostics.js" defer></script>', '')
    page = page.replace('<link rel="stylesheet" href="/workspace.css">', '<style>' + css + '</style>')
    page = page.replace('<script src="/workspace.js" defer></script>', '<script>window.RESEARCH_DATA=' + data + ';</script><script defer>' + js + '</script>')
    # Inline scripts are not deferred by browsers. Run the UI after its DOM exists.
    first = page.index('<script>window.RESEARCH_DATA=')
    last = page.index('</script>', page.index('<script defer>', first)) + len('</script>')
    scripts = page[first:last].replace('<script defer>', '<script>')
    page = page[:first] + page[last:]
    page = page.replace('</body>', scripts + '</body>')
    (destination / "index.html").write_text(page, encoding="utf-8")
    (destination / "index.json").write_text(json.dumps(catalog, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(f"Offline research dashboard: {len(catalog['files'])} artifacts / {len(catalog['runs'])} runs")


if __name__ == "__main__":
    main()
