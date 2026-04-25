from __future__ import annotations

import re

from rag_template_core.domain.models import Document, DocumentNode


class MarkdownSkeletonParser:
    _heading_pattern = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

    def parse(self, *, doc_id: str, markdown: str) -> Document:
        lines = markdown.splitlines()
        heading_indices = [
            (index, match)
            for index, line in enumerate(lines)
            if (match := self._heading_pattern.match(line))
        ]
        if not heading_indices:
            return Document(
                doc_id=doc_id,
                nodes=(
                    DocumentNode(
                        doc_id=doc_id,
                        node_id=f"{doc_id}:root",
                        title=doc_id,
                        level=0,
                        order=1,
                        content=markdown,
                        parent_id=None,
                        breadcrumb=(doc_id,),
                    ),
                ),
            )

        stack: list[DocumentNode] = []
        nodes: list[DocumentNode] = []
        for order, (line_index, match) in enumerate(heading_indices, start=1):
            level = len(match.group(1))
            title = match.group(2).strip()
            next_line_index = (
                heading_indices[order][0] if order < len(heading_indices) else len(lines)
            )
            content_lines = lines[line_index + 1 : next_line_index]
            if order == 1 and line_index > 0:
                content_lines = [*lines[:line_index], *content_lines]
            content = "\n".join(content_lines).strip()

            while stack and stack[-1].level >= level:
                stack.pop()

            parent = stack[-1] if stack else None
            breadcrumb = (*(parent.breadcrumb if parent else ()), title)
            node = DocumentNode(
                doc_id=doc_id,
                node_id=f"{doc_id}:n{order}",
                title=title,
                level=level,
                order=order,
                content=content,
                parent_id=parent.node_id if parent else None,
                breadcrumb=breadcrumb,
            )
            nodes.append(node)
            stack.append(node)

        return Document(doc_id=doc_id, nodes=tuple(nodes))
