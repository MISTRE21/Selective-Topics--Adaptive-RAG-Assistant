import re
from pathlib import Path

report_path = Path("REPORT.md")
content = report_path.read_text(encoding="utf-8")

# Convert basic markdown elements to HTML
html_body = []
lines = content.splitlines()
in_code = False
code_lang = ""
code_buffer = []

for line in lines:
    if line.startswith("```"):
        if in_code:
            code_str = "\n".join(code_buffer)
            html_body.append(f"<pre><code>{code_str}</code></pre>")
            code_buffer = []
            in_code = False
        else:
            in_code = True
            code_lang = line[3:].strip()
            code_buffer = []
        continue

    if in_code:
        code_buffer.append(line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
        continue

    # Headers
    if line.startswith("# "):
        html_body.append(f"<h1>{line[2:].strip()}</h1>")
    elif line.startswith("## "):
        html_body.append(f"<h2>{line[3:].strip()}</h2>")
    elif line.startswith("### "):
        html_body.append(f"<h3>{line[4:].strip()}</h3>")
    elif line.startswith("#### "):
        html_body.append(f"<h4>{line[5:].strip()}</h4>")
    elif line.startswith("|") and line.endswith("|"):
        # Table row
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if all(re.match(r"^:?-+:?$", c) for c in cells):
            continue
        row_html = "".join(f"<td>{c}</td>" for c in cells)
        html_body.append(f"<tr>{row_html}</tr>")
    elif line.strip() == "---":
        html_body.append("<hr/>")
    elif line.startswith("- "):
        html_body.append(f"<li>{line[2:].strip()}</li>")
    elif line.strip():
        # Text formatting
        txt = line
        txt = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", txt)
        txt = re.sub(r"\*(.*?)\*", r"<em>\1</em>", txt)
        txt = re.sub(r"`(.*?)`", r"<code>\1</code>", txt)
        html_body.append(f"<p>{txt}</p>")

full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Adaptive RAG Assistant for Software Project Documentation</title>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    line-height: 1.6;
    color: #24292e;
    max-width: 900px;
    margin: 40px auto;
    padding: 0 20px;
  }}
  h1 {{ font-size: 2.2rem; border-bottom: 2px solid #eaecef; padding-bottom: 12px; margin-bottom: 8px; color: #1a202c; }}
  h2 {{ font-size: 1.6rem; border-bottom: 1px solid #eaecef; padding-bottom: 8px; margin-top: 32px; color: #2d3748; }}
  h3 {{ font-size: 1.25rem; margin-top: 24px; color: #4a5568; }}
  table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
  th, td {{ border: 1px solid #cbd5e0; padding: 10px 14px; text-align: left; }}
  tr:nth-child(even) {{ background-color: #f7fafc; }}
  pre {{ background: #2d3748; color: #f7fafc; padding: 16px; border-radius: 6px; overflow-x: auto; font-family: Consolas, monospace; }}
  code {{ background: #edf2f7; color: #b83280; padding: 2px 6px; border-radius: 4px; font-size: 0.9em; }}
  pre code {{ background: none; color: inherit; padding: 0; }}
  blockquote {{ border-left: 4px solid #3182ce; padding-left: 16px; margin: 16px 0; color: #4a5568; }}
  li {{ margin-bottom: 6px; }}
  hr {{ border: none; border-top: 1px solid #e2e8f0; margin: 30px 0; }}
</style>
</head>
<body>
{"".join(html_body)}
</body>
</html>
"""

Path("REPORT.html").write_text(full_html, encoding="utf-8")
print("Successfully generated REPORT.html")
