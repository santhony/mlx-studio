// chat-render.js — shared sentinel renderer for chat-like content.
//
// Sentinels recognized in model output:
//   <think>...</think>             reasoning / chain of thought
//   <tool name="...">JSON</tool>   the model called a tool with these args
//   <tool_result>...</tool_result> the result returned for the previous call
//
// Anywhere a model reply may be rendered (Chat, RAG chat, Notebook), call
// window.renderChatContent(rawText) and assign the returned HTML into the
// target element via innerHTML. The renderer escapes HTML so untrusted
// model text cannot inject script tags.
//
// A separate helper, window.splitThinkAndBody(rawText), partitions text
// into { thinking: string, body: string } and is useful when the body
// itself needs further processing (e.g. Prism syntax highlighting in
// notebook cells, where the thinking block must live outside <pre><code>).
(function () {
    function esc(s) {
        return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    }

    // One regex with four alternations, longest tag names first so e.g.
    // "tool_result" beats a shorter "tool" alternative. The second branch
    // handles a still-streaming <think> with no closing tag yet.
    var SENTINEL_RE = /<think>([\s\S]*?)<\/think>|<think>([\s\S]*)$|<tool_result>([\s\S]*?)<\/tool_result>|<tool name="([^"]*)">([\s\S]*?)<\/tool>/g;

    window.renderChatContent = function (raw) {
        if (!raw) return "";
        var out = "";
        var last = 0;
        var m;
        SENTINEL_RE.lastIndex = 0;
        while ((m = SENTINEL_RE.exec(raw)) !== null) {
            if (m.index > last) out += esc(raw.slice(last, m.index));
            if (m[1] !== undefined) {
                out += '<span class="thinking">' + esc(m[1]) + '</span>';
            } else if (m[2] !== undefined) {
                out += '<span class="thinking">' + esc(m[2]) + '</span>';
            } else if (m[3] !== undefined) {
                out += '<details class="tool-result"><summary>tool result</summary><pre>' + esc(m[3]) + '</pre></details>';
            } else if (m[4] !== undefined) {
                out += '<details class="tool-call" open><summary>called <code>' + esc(m[4]) + '</code></summary><pre>' + esc(m[5]) + '</pre></details>';
            }
            last = SENTINEL_RE.lastIndex;
            if (m[2] !== undefined) { last = raw.length; break; }
        }
        if (last < raw.length) out += esc(raw.slice(last));
        return out;
    };

    // For consumers that need to render thinking elsewhere and keep the
    // body as plain text (e.g. notebook code cells that feed Prism).
    // Concatenates any text outside <think>...</think> as the body and any
    // text inside as the thinking. Tool sentinels are kept in the body so
    // they still render via renderChatContent if applied.
    window.splitThinkAndBody = function (raw) {
        if (!raw) return { thinking: "", body: "" };
        var thinking = "";
        var body = "";
        var i = 0;
        while (i < raw.length) {
            var open = raw.indexOf("<think>", i);
            if (open === -1) { body += raw.slice(i); break; }
            if (open > i) body += raw.slice(i, open);
            var close = raw.indexOf("</think>", open + 7);
            if (close === -1) { thinking += raw.slice(open + 7); break; }
            thinking += raw.slice(open + 7, close);
            i = close + 8;
        }
        return { thinking: thinking, body: body };
    };

    // Auto-render historic content on any page that opts in by marking
    // .msg-content elements with data-raw. Idempotent — running twice on
    // the same element only renders once (innerHTML overwrite is fine).
    function renderHistoricMessages(root) {
        (root || document).querySelectorAll(".msg-content[data-raw]").forEach(function (el) {
            // textContent gives us the raw text the browser unescaped from
            // the server-rendered HTML — exactly what we want to feed to
            // the sentinel parser.
            el.innerHTML = window.renderChatContent(el.textContent);
            el.removeAttribute("data-raw");
        });
    }
    document.addEventListener("DOMContentLoaded", function () { renderHistoricMessages(document); });
    // HTMX swaps don't re-fire DOMContentLoaded; re-run on any swap so
    // newly inserted historic messages (none today, but safe for future use)
    // still get rendered.
    document.addEventListener("htmx:afterSwap", function (evt) { renderHistoricMessages(evt.detail.target); });
})();
