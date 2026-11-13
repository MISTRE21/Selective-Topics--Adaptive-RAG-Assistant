import re
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

@dataclass
class Chunk:
    chunk_id: str
    doc_name: str
    section_title: str
    content: str
    char_length: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chunk":
        return cls(**data)


def slugify(text: str) -> str:
    """Normalize header text into an anchor slug."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    return text.strip("-")


class DocumentLoader:
    def __init__(self, raw_docs_dir: Path):
        self.raw_docs_dir = Path(raw_docs_dir)

    def load_and_chunk(self) -> List[Chunk]:
        """Load all markdown files and split into semantic chunks based on headers."""
        chunks: List[Chunk] = []
        md_files = sorted(self.raw_docs_dir.glob("*.md"))
        
        for md_file in md_files:
            text = md_file.read_text(encoding="utf-8")
            file_chunks = self.chunk_markdown(md_file.name, text)
            chunks.extend(file_chunks)
            
        return chunks

    def chunk_markdown(self, filename: str, text: str) -> List[Chunk]:
        """Split a markdown file into semantic chunks anchored on headers."""
        lines = text.splitlines()
        chunks: List[Chunk] = []
        
        current_doc_title = filename
        current_section = "Overview"
        current_slug = "overview"
        buffer: List[str] = []

        header_pattern = re.compile(r"^(#{1,3})\s+(.*)$")

        for line in lines:
            match = header_pattern.match(line.strip())
            if match:
                level = len(match.group(1))
                title = match.group(2).strip()

                if level == 1:
                    current_doc_title = title
                    
                    if buffer:
                        content_str = "\n".join(buffer).strip()
                        if content_str:
                            chunk_id = f"{filename}#{current_slug}"
                            chunks.append(Chunk(
                                chunk_id=chunk_id,
                                doc_name=filename,
                                section_title=f"{current_doc_title} - {current_section}",
                                content=content_str,
                                char_length=len(content_str)
                            ))
                        buffer = []
                elif level in (2, 3):
                    
                    if buffer:
                        content_str = "\n".join(buffer).strip()
                        if content_str:
                            chunk_id = f"{filename}#{current_slug}"
                            chunks.append(Chunk(
                                chunk_id=chunk_id,
                                doc_name=filename,
                                section_title=f"{current_doc_title} - {current_section}",
                                content=content_str,
                                char_length=len(content_str)
                            ))
                    buffer = [f"### Context: {current_doc_title} > {title}\n"]
                    current_section = title
                    current_slug = slugify(title)
            else:
                buffer.append(line)

        
        if buffer:
            content_str = "\n".join(buffer).strip()
            if content_str:
                chunk_id = f"{filename}#{current_slug}"
                chunks.append(Chunk(
                    chunk_id=chunk_id,
                    doc_name=filename,
                    section_title=f"{current_doc_title} - {current_section}",
                    content=content_str,
                    char_length=len(content_str)
                ))

        return chunks

    def save_processed_chunks(self, chunks: List[Chunk], dest_path: Path):
        dest_path = Path(dest_path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, "w", encoding="utf-8") as f:
            json.dump([c.to_dict() for c in chunks], f, indent=2)

    @staticmethod
    def load_processed_chunks(src_path: Path) -> List[Chunk]:
        with open(src_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return [Chunk.from_dict(item) for item in raw]
