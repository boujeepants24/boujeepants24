"use client";

import { useEffect, useRef } from "react";
import { ChevronDown, Terminal } from "lucide-react";
import type { LogTag } from "@/lib/types";
import { cn } from "@/lib/utils";

export interface LogLine {
  id: number;
  tag: LogTag;
  message: string;
  ts: number;
}

const TAG_STYLES: Record<LogTag, string> = {
  SYSTEM: "text-cyan-400",
  SEARCH: "text-sky-400",
  CRAWLER: "text-violet-400",
  "GROQ-70B": "text-emerald-400",
  CACHE: "text-teal-300",
  WARN: "text-amber-400",
  ERROR: "text-rose-400",
  DONE: "text-emerald-300",
};

interface Props {
  lines: LogLine[];
  running: boolean;
  open: boolean;
  onToggle: () => void;
}

export function TerminalLog({ lines, running, open, onToggle }: Props) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const startTs = lines[0]?.ts ?? 0;

  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [lines.length, open]);

  return (
    <section className="overflow-hidden rounded-lg border border-line bg-[#0b0c10]">
      <button
        type="button"
        onClick={onToggle}
        className="flex w-full items-center justify-between gap-3 border-b border-line bg-panel px-4 py-2 text-left"
        aria-expanded={open}
      >
        <span className="flex items-center gap-2 font-mono text-[11px] uppercase tracking-[0.18em] text-muted">
          <Terminal className="size-3.5 text-emerald-400" />
          Live Terminal
          <span
            className={cn(
              "ml-1 inline-flex items-center gap-1.5 rounded border px-1.5 py-0.5 text-[10px] tracking-wider",
              running ? "border-emerald-500/30 text-emerald-400" : "border-line text-faint",
            )}
          >
            <span className={cn("size-1.5 rounded-full", running ? "animate-pulse bg-emerald-400" : "bg-faint")} />
            {running ? "STREAMING" : lines.length ? "IDLE" : "STANDBY"}
          </span>
        </span>
        <ChevronDown className={cn("size-4 text-muted transition-transform", open && "rotate-180")} />
      </button>

      {open && (
        <div ref={scrollRef} className="max-h-56 overflow-y-auto px-4 py-3 font-mono text-[12px] leading-relaxed">
          {lines.length === 0 ? (
            <p className="text-faint">
              <span className="text-emerald-500">vanguard@radar</span>:~$ awaiting sweep command
              <span className="animate-caret ml-0.5 inline-block h-3.5 w-1.5 translate-y-0.5 bg-emerald-400" />
            </p>
          ) : (
            lines.map((line) => (
              <div key={line.id} className="animate-fade-in flex gap-3 whitespace-pre-wrap break-words">
                <span className="shrink-0 tabular-nums text-faint">+{((line.ts - startTs) / 1000).toFixed(2)}s</span>
                <span className={cn("shrink-0", TAG_STYLES[line.tag])}>[{line.tag}]</span>
                <span className="min-w-0 text-ink/85">{line.message}</span>
              </div>
            ))
          )}
          {running && (
            <div className="mt-1 flex gap-3 text-faint">
              <span className="animate-caret inline-block h-3.5 w-1.5 translate-y-0.5 bg-emerald-400" />
            </div>
          )}
        </div>
      )}
    </section>
  );
}
