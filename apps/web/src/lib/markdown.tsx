import { Fragment, type ReactNode } from "react";

/**
 * Tiny purpose-built markdown renderer for AI Ops chat output.
 *
 * Handles:
 *   - paragraphs / blank-line separation
 *   - ATX headings  # ## ### ####
 *   - unordered lists `- item`, `* item`
 *   - ordered lists `1. item`
 *   - fenced code blocks ``` ... ```
 *   - inline: **bold**, *italic*, `code`, ~~strike~~
 *
 * Escapes raw HTML by virtue of building React nodes from strings.
 * No external dependency.
 */
export function MarkdownText({ text, onLink }: { text: string; onLink?: (href: string) => boolean }): JSX.Element {
  if (!text) return <Fragment />;
  const safeText = sanitizeModelProtocol(text);
  if (!safeText) return <Fragment />;
  const blocks = parseBlocks(safeText);
  return (
    <div className="md-text">
      {blocks.map((block, i) => renderBlock(block, i, onLink))}
    </div>
  );
}

// Never render legacy tool markup as user-facing Markdown. The backend rejects
// it and retries, but this guard protects older runs and streamed responses.
function sanitizeModelProtocol(input: string): string {
  const lower = input.toLowerCase();
  if (!lower.includes("<toolcall") && !lower.includes("<function=") && !lower.includes("<parameter=")) return input;
  return "工具调用协议格式错误，系统正在重新请求模型。";
}

type Block =
  | { kind: "para"; text: string }
  | { kind: "heading"; level: number; text: string }
  | { kind: "ul"; items: string[] }
  | { kind: "ol"; items: string[] }
  | { kind: "table"; headers: string[]; rows: string[][] }
  | { kind: "code"; lang: string; text: string };

function parseBlocks(input: string): Block[] {
  const lines = input.replace(/\r\n?/g, "\n").split("\n");
  const blocks: Block[] = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];

    if (!line.trim()) {
      i++;
      continue;
    }

    // Fenced code
    const fence = /^```(\w*)\s*$/.exec(line);
    if (fence) {
      const lang = fence[1] || "";
      i++;
      const codeLines: string[] = [];
      while (i < lines.length && !/^```/.test(lines[i])) {
        codeLines.push(lines[i]);
        i++;
      }
      if (i < lines.length) i++; // skip closing fence
      blocks.push({ kind: "code", lang, text: codeLines.join("\n") });
      continue;
    }

    // Heading
    const heading = /^(#{1,4})\s+(.+)$/.exec(line);
    if (heading) {
      blocks.push({ kind: "heading", level: heading[1].length, text: heading[2].trim() });
      i++;
      continue;
    }

    // GitHub-style table. The separator row is structural and is not shown.
    if (i + 1 < lines.length && lines[i].includes("|") && isTableSeparator(lines[i + 1])) {
      const headers = splitTableRow(lines[i]);
      i += 2;
      const rows: string[][] = [];
      while (i < lines.length && lines[i].trim() && lines[i].includes("|") && !isBlockStart(lines[i])) {
        rows.push(splitTableRow(lines[i]));
        i++;
      }
      blocks.push({ kind: "table", headers, rows });
      continue;
    }

    // Unordered list
    if (/^\s*[-*]\s+/.test(line)) {
      const items: string[] = [];
      while (i < lines.length && /^\s*[-*]\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^\s*[-*]\s+/, ""));
        i++;
      }
      blocks.push({ kind: "ul", items });
      continue;
    }

    // Ordered list
    if (/^\s*\d+\.\s+/.test(line)) {
      const items: string[] = [];
      while (i < lines.length && /^\s*\d+\.\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^\s*\d+\.\s+/, ""));
        i++;
      }
      blocks.push({ kind: "ol", items });
      continue;
    }

    // Paragraph (collect consecutive non-blank lines)
    const buf: string[] = [line];
    i++;
    while (i < lines.length && lines[i].trim() && !isBlockStart(lines[i])) {
      buf.push(lines[i]);
      i++;
    }
    blocks.push({ kind: "para", text: buf.join(" ") });
  }

  return blocks;
}

function splitTableRow(line: string): string[] {
  return line.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((cell) => cell.trim());
}

function isTableSeparator(line: string): boolean {
  const cells = splitTableRow(line);
  return cells.length > 0 && cells.every((cell) => /^:?-{3,}:?$/.test(cell));
}

function isBlockStart(line: string): boolean {
  return (
    /^```/.test(line) ||
    /^#{1,4}\s+/.test(line) ||
    /^\s*[-*]\s+/.test(line) ||
    /^\s*\d+\.\s+/.test(line)
  );
}

function renderBlock(block: Block, key: number, onLink?: (href: string) => boolean): JSX.Element {
  switch (block.kind) {
    case "code":
      return (
        <pre className="md-code" key={key}>
          <code>{block.text}</code>
        </pre>
      );
    case "heading": {
      const Tag = (`h${Math.min(block.level + 2, 6)}` as keyof JSX.IntrinsicElements);
      return <Tag className={`md-heading md-h${block.level}`} key={key}>{renderInline(block.text, onLink)}</Tag>;
    }
    case "ul":
      return (
        <ul className="md-ul" key={key}>
          {block.items.map((it, idx) => (
            <li key={idx}>{renderInline(it, onLink)}</li>
          ))}
        </ul>
      );
    case "ol":
      return (
        <ol className="md-ol" key={key}>
          {block.items.map((it, idx) => (
            <li key={idx}>{renderInline(it, onLink)}</li>
          ))}
        </ol>
      );
    case "table":
      return (
        <div className="md-table-scroll" key={key}>
          <table className="md-table">
            <thead><tr>{block.headers.map((cell, index) => <th key={index}>{renderInline(cell, onLink)}</th>)}</tr></thead>
            <tbody>{block.rows.map((row, rowIndex) => <tr key={rowIndex}>{block.headers.map((_, cellIndex) => <td key={cellIndex}>{renderInline(row[cellIndex] || "", onLink)}</td>)}</tr>)}</tbody>
          </table>
        </div>
      );
    case "para":
      return <p className="md-p" key={key}>{renderInline(block.text, onLink)}</p>;
  }
}

/**
 * Inline tokenizer. Walks the string and matches `**bold**`, `*italic*`,
 * `` `code` ``, and `~~strike~~`. Everything else stays as plain text.
 */
function renderInline(text: string, onLink?: (href: string) => boolean): ReactNode {
  const tokens: ReactNode[] = [];
  let rest = text;
  let key = 0;

  const patterns: { regex: RegExp; render: (match: RegExpExecArray) => ReactNode }[] = [
    { regex: /^\[([^\]]+)\]\(([^\s)]+)(?:\s+"[^"]*")?\)/, render: (m) => {
      const href = safeLink(m[2]);
      if (!href) return <span key={key++}>{m[1]}</span>;
      return <a className="md-link" href={href} key={key++} target={href.startsWith("http") ? "_blank" : undefined} rel="noreferrer" onClick={(event) => {
        if (onLink?.(href)) event.preventDefault();
      }}>{m[1]}</a>;
    } },
    { regex: /^`([^`]+)`/, render: (m) => <code className="md-inline-code" key={key++}>{m[1]}</code> },
    { regex: /^\*\*([^*]+)\*\*/, render: (m) => <strong key={key++}>{m[1]}</strong> },
    { regex: /^__([^_]+)__/, render: (m) => <strong key={key++}>{m[1]}</strong> },
    { regex: /^\*([^*\s][^*]*?)\*/, render: (m) => <em key={key++}>{m[1]}</em> },
    { regex: /^_([^_\s][^_]*?)_/, render: (m) => <em key={key++}>{m[1]}</em> },
    { regex: /^~~([^~]+)~~/, render: (m) => <s key={key++}>{m[1]}</s> }
  ];

  while (rest.length > 0) {
    let matched = false;
    for (const { regex, render } of patterns) {
      const m = regex.exec(rest);
      if (m) {
        tokens.push(render(m));
        rest = rest.slice(m[0].length);
        matched = true;
        break;
      }
    }
    if (!matched) {
      // Eat a single char (or up to the next special). Keep batches as plain text.
      const nextSpecial = rest.search(/[\[*_`~]/);
      const chunkLen = nextSpecial === -1 ? rest.length : Math.max(1, nextSpecial);
      tokens.push(rest.slice(0, chunkLen));
      rest = rest.slice(chunkLen);
    }
  }
  return tokens;
}

function safeLink(value: string): string | undefined {
  const href = value.trim();
  if (/^(?:https?:|mailto:)/i.test(href) || /^(?:\/|\.\.?\/|#)/.test(href) || /^[\w@.+-][\w@.+/ -]*$/u.test(href)) return href;
  return undefined;
}
