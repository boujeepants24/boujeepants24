"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ArrowRight, Loader2 } from "lucide-react";
import { DEFAULT_PRESET, PRESETS, getPreset } from "@/lib/presets";
import { buildCampaignCsv, downloadCsv, type LeadDraft } from "@/lib/csv";
import type { Lead, PresetId, SweepEvent, SweepMode } from "@/lib/types";
import { cn } from "@/lib/utils";
import { InspectorDrawer } from "./inspector-drawer";
import { ResultsTable } from "./results-table";
import { TerminalLog, type LogLine } from "./terminal-log";
import { CountUp, formatEur, parseEur } from "./ui";

export function RadarConsole() {
  const [presetId, setPresetId] = useState<PresetId>(DEFAULT_PRESET.id);
  const [query, setQuery] = useState(DEFAULT_PRESET.query);
  const [offer, setOffer] = useState(DEFAULT_PRESET.offer);
  const [mode, setMode] = useState<SweepMode>("live");

  const [running, setRunning] = useState(false);
  const [logs, setLogs] = useState<LogLine[]>([]);
  const [terminalOpen, setTerminalOpen] = useState(true);
  const [leads, setLeads] = useState<Lead[]>([]);
  const [source, setSource] = useState<"live" | "cache" | null>(null);
  const [fallbackReason, setFallbackReason] = useState<string | undefined>();
  const [drafts, setDrafts] = useState<Record<string, LeadDraft>>({});
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [toast, setToast] = useState<string | null>(null);

  const logId = useRef(0);
  const abortRef = useRef<AbortController | null>(null);
  const toastTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const showToast = useCallback((msg: string) => {
    setToast(msg);
    if (toastTimer.current) clearTimeout(toastTimer.current);
    toastTimer.current = setTimeout(() => setToast(null), 2400);
  }, []);

  useEffect(() => () => abortRef.current?.abort(), []);

  const pushLog = useCallback((tag: LogLine["tag"], message: string, ts = Date.now()) => {
    setLogs((prev) => [...prev, { id: logId.current++, tag, message, ts }]);
  }, []);

  const applyPreset = (id: PresetId) => {
    setPresetId(id);
    const p = getPreset(id);
    if (id !== "custom") {
      setQuery(p.query);
      setOffer(p.offer);
    }
  };

  const handleEvent = useCallback(
    (event: SweepEvent) => {
      if (event.type === "log") pushLog(event.tag, event.message, event.ts);
      else if (event.type === "error") pushLog("ERROR", event.message, event.ts);
      else {
        setLeads(event.leads);
        setSource(event.source);
        setFallbackReason(event.fallbackReason);
        setDrafts(
          Object.fromEntries(
            event.leads.map((l) => [
              l.id,
              { subject: l.sniperSubject, body: l.sniperEmail, firstName: "", lastName: "", pushed: false },
            ]),
          ),
        );
      }
    },
    [pushLog],
  );

  const runSweep = async () => {
    if (running || !query.trim() || !offer.trim()) return;
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setRunning(true);
    setTerminalOpen(true);
    setLogs([]);
    setSelectedId(null);

    try {
      const res = await fetch("/api/sweep", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, offer, mode, presetId }),
        signal: controller.signal,
      });
      if (!res.ok || !res.body) {
        const err = (await res.json().catch(() => null)) as { error?: string } | null;
        throw new Error(err?.error ?? `Sweep failed (HTTP ${res.status})`);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";
        for (const line of lines) {
          if (line.trim()) handleEvent(JSON.parse(line) as SweepEvent);
        }
      }
      if (buffer.trim()) handleEvent(JSON.parse(buffer) as SweepEvent);
    } catch (err) {
      if (controller.signal.aborted) return;
      pushLog("ERROR", err instanceof Error ? err.message : "Sweep failed");
    } finally {
      if (abortRef.current === controller) setRunning(false);
    }
  };

  const updateDraft = useCallback((id: string, patch: Partial<LeadDraft>) => {
    setDrafts((prev) => ({ ...prev, [id]: { ...prev[id], ...patch } }));
  }, []);

  const exportCsv = () => {
    if (leads.length === 0) return;
    const stamp = new Date().toISOString().slice(0, 16).replace(/[:T]/g, "-");
    downloadCsv(`vanguard-campaign-${stamp}.csv`, buildCampaignCsv(leads, drafts));
    showToast(`Exported ${leads.length} leads to CSV`);
  };

  const stats = useMemo(() => {
    const pipeline = leads.reduce((sum, l) => sum + parseEur(l.estimatedAnnualValue), 0);
    const avgFit = leads.length ? leads.reduce((s, l) => s + l.fitScore, 0) / leads.length : 0;
    const queued = Object.values(drafts).filter((d) => d.pushed).length;
    return { pipeline, avgFit, queued };
  }, [leads, drafts]);

  const selectedIndex = leads.findIndex((l) => l.id === selectedId);
  const selected = selectedIndex >= 0 ? leads[selectedIndex] : null;
  const closeInspector = useCallback(() => setSelectedId(null), []);
  const prevLead = selectedIndex > 0 ? () => setSelectedId(leads[selectedIndex - 1].id) : undefined;
  const nextLead =
    selectedIndex >= 0 && selectedIndex < leads.length - 1 ? () => setSelectedId(leads[selectedIndex + 1].id) : undefined;

  return (
    <div className="min-h-screen">
      {/* Top header */}
      <header className="sticky top-0 z-40 border-b border-line bg-obsidian/95 backdrop-blur">
        <div className="mx-auto flex max-w-[1440px] flex-wrap items-center justify-between gap-3 px-4 py-3.5 sm:px-8">
          <div className="flex items-center gap-3.5">
            <Monogram running={running} />
            <div className="text-[13px] font-semibold tracking-[0.22em] text-ink">
              VANGUARD <span className="font-normal text-faint">{"//"}</span> SIGNAL RADAR{" "}
              <span className="ml-1 font-mono text-[11px] font-normal tracking-normal text-brass">v2.4</span>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-5">
            <label className="flex items-center gap-2.5">
              <span className="hidden text-[11px] uppercase tracking-[0.14em] text-faint md:inline">Preset</span>
              <select
                value={presetId}
                onChange={(e) => applyPreset(e.target.value as PresetId)}
                className="max-w-[82vw] rounded border border-line-strong bg-panel px-2.5 py-1.5 text-xs text-ink transition focus:border-brass focus:outline-none sm:max-w-none"
              >
                {PRESETS.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.label}
                  </option>
                ))}
              </select>
            </label>
            <div className="hidden items-center gap-5 border-l border-line pl-5 text-[11px] text-faint lg:flex">
              <span>
                Campaign queue{" "}
                <span className="ml-1 font-mono text-[12px] tabular-nums text-ink">{stats.queued}</span>
              </span>
              <UtcClock />
            </div>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-[1440px] space-y-5 px-4 py-7 sm:px-8">
        <div className="flex flex-wrap items-end justify-between gap-2">
          <div>
            <h1 className="font-serif text-[26px] leading-tight text-ink">Buyer Signal Sweep</h1>
            <p className="mt-1 text-[13px] text-muted">
              Identify in-market hardware buyers from their own websites and draft procurement outreach.
            </p>
          </div>
        </div>

        {/* Control panel */}
        <section className="rounded-md border border-line bg-panel">
          <div className="grid gap-px bg-line lg:grid-cols-[1fr_1.4fr]">
            <Field index="01" label="Target Buyer Signal / Niche Search">
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && runSweep()}
                placeholder="industrial IoT hardware manufacturers Germany UK"
                maxLength={200}
                className="w-full border-0 border-b border-line-strong bg-transparent px-0 py-2 text-[14px] text-ink transition placeholder:text-faint focus:border-brass focus:outline-none"
              />
            </Field>
            <Field index="02" label="Client Offer & Capability">
              <textarea
                value={offer}
                onChange={(e) => {
                  setOffer(e.target.value);
                  if (presetId !== "custom") setPresetId("custom");
                }}
                rows={2}
                maxLength={800}
                placeholder="What you sell, with the concrete cost / speed / quality edge"
                className="w-full resize-none border-0 border-b border-line-strong bg-transparent px-0 py-2 text-[14px] leading-relaxed text-ink transition placeholder:text-faint focus:border-brass focus:outline-none"
              />
            </Field>
          </div>

          <div className="flex flex-wrap items-center justify-between gap-4 border-t border-line px-5 py-4">
            <div role="radiogroup" aria-label="Sweep mode" className="flex items-center gap-1">
              <span className="mr-2 text-[11px] uppercase tracking-[0.14em] text-faint">Mode</span>
              <ModeButton active={mode === "live"} onClick={() => setMode("live")}>
                Live Web Scrape + Groq AI
              </ModeButton>
              <ModeButton active={mode === "cache"} onClick={() => setMode("cache")}>
                Cached Instant Demo
              </ModeButton>
            </div>

            <button
              type="button"
              onClick={runSweep}
              disabled={running || !query.trim() || !offer.trim()}
              className="group inline-flex items-center gap-2.5 rounded bg-ivory px-5 py-2.5 text-[13px] font-semibold text-obsidian transition hover:bg-white disabled:cursor-not-allowed disabled:opacity-50"
            >
              {running && <Loader2 className="size-4 animate-spin" />}
              {running ? "Sweep in progress" : "Execute Signal Sweep"}
              {!running && <ArrowRight className="size-4 transition-transform group-hover:translate-x-0.5" />}
            </button>
          </div>
        </section>

        <TerminalLog lines={logs} running={running} open={terminalOpen} onToggle={() => setTerminalOpen((o) => !o)} />

        {leads.length > 0 && (
          <div className="grid grid-cols-2 gap-px overflow-hidden rounded-md border border-line bg-line md:grid-cols-4">
            <Stat label="Qualified Buyers">
              <CountUp value={leads.length} format={(n) => String(Math.round(n))} />
            </Stat>
            <Stat label="Average Fit Score">
              <CountUp value={stats.avgFit} format={(n) => n.toFixed(1)} />
            </Stat>
            <Stat label="Pipeline Value" accent>
              <CountUp value={stats.pipeline} format={(n) => formatEur(n)} />
              <span className="ml-1 text-sm text-faint">/yr</span>
            </Stat>
            <Stat label="Queued for Outbound">
              {stats.queued}
              <span className="text-faint"> / {leads.length}</span>
            </Stat>
          </div>
        )}

        {leads.length > 0 ? (
          <ResultsTable
            leads={leads}
            drafts={drafts}
            selectedId={selectedId}
            onSelect={setSelectedId}
            onExport={exportCsv}
            source={source}
            fallbackReason={fallbackReason}
          />
        ) : (
          <EmptyState running={running} />
        )}

        <footer className="flex flex-wrap justify-between gap-2 border-t border-line pt-4 text-[11px] text-faint">
          <span>Vanguard Signal Radar · Buyer intelligence & outbound</span>
          <span>Sources: DuckDuckGo HTML index · Company websites · Groq llama-3.3-70b-versatile</span>
        </footer>
      </main>

      {selected && drafts[selected.id] && (
        <InspectorDrawer
          key={selected.id}
          lead={selected}
          rank={selectedIndex + 1}
          total={leads.length}
          draft={drafts[selected.id]}
          onChange={(patch) => updateDraft(selected.id, patch)}
          onClose={closeInspector}
          onPrev={prevLead}
          onNext={nextLead}
          onToast={showToast}
        />
      )}

      {toast && (
        <div
          role="status"
          className="animate-fade-in fixed bottom-6 left-1/2 z-[60] flex -translate-x-1/2 items-center gap-2.5 rounded border border-line-strong bg-panel-2 px-4 py-2.5 text-xs text-ink shadow-xl shadow-black/50"
        >
          <span className="size-1.5 rounded-full bg-brass" />
          {toast}
        </div>
      )}
    </div>
  );
}

/** Square monogram; a brass sweep line rotates inside it while a sweep runs. */
function Monogram({ running }: { running: boolean }) {
  return (
    <div className="relative grid size-8 place-items-center overflow-hidden rounded-sm border border-line-strong bg-panel">
      <span className="font-serif text-[15px] font-semibold text-ink">V</span>
      <span
        className={cn(
          "absolute left-1/2 top-1/2 h-px w-1/2 origin-left bg-brass transition-opacity",
          running ? "animate-[spin_1.6s_linear_infinite] opacity-80" : "opacity-0",
        )}
      />
      <span className="absolute inset-x-0 bottom-0 h-[2px] bg-brass" />
    </div>
  );
}

function UtcClock() {
  const [now, setNow] = useState<Date | null>(null);
  useEffect(() => {
    const tick = () => setNow(new Date());
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, []);
  return (
    <span className="font-mono text-[12px] tabular-nums text-muted" suppressHydrationWarning>
      {now ? `${now.toISOString().slice(11, 19)} UTC` : "--:--:-- UTC"}
    </span>
  );
}

function Field({ index, label, children }: { index: string; label: string; children: React.ReactNode }) {
  return (
    <label className="block bg-panel px-5 pb-4 pt-4">
      <span className="mb-1 flex items-baseline gap-2 text-[11px] uppercase tracking-[0.14em] text-muted">
        <span className="font-mono text-brass">{index}</span>
        {label}
      </span>
      {children}
    </label>
  );
}

function ModeButton({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      type="button"
      role="radio"
      aria-checked={active}
      onClick={onClick}
      className={cn(
        "flex items-center gap-2 rounded border px-3 py-1.5 text-xs transition",
        active ? "border-line-strong bg-panel-2 text-ink" : "border-transparent text-faint hover:text-muted",
      )}
    >
      <span
        className={cn(
          "grid size-3 place-items-center rounded-full border transition-colors",
          active ? "border-brass" : "border-line-strong",
        )}
      >
        <span className={cn("size-1.5 rounded-full transition-transform", active ? "scale-100 bg-brass" : "scale-0")} />
      </span>
      {children}
    </button>
  );
}

function Stat({ label, accent, children }: { label: string; accent?: boolean; children: React.ReactNode }) {
  return (
    <div className="animate-fade-in bg-panel px-5 py-4">
      <div className="text-[11px] uppercase tracking-[0.14em] text-faint">{label}</div>
      <div className={cn("mt-1.5 font-serif text-[26px] leading-none tabular-nums", accent ? "text-brass" : "text-ink")}>
        {children}
      </div>
    </div>
  );
}

function EmptyState({ running }: { running: boolean }) {
  return (
    <section className="rounded-md border border-line bg-panel px-6 py-14">
      <div className="mx-auto max-w-xl text-center">
        <p className="font-serif text-xl text-ink">{running ? "Sweep in progress" : "No buyers identified yet"}</p>
        <p className="mt-2 text-[13px] leading-relaxed text-muted">
          Choose a preset or describe the niche and offer, then execute a sweep. Live mode reads real company websites and
          drafts outreach with Groq. The instant cache returns a pre-qualified dataset in 1.5 seconds.
        </p>
        <div className="mx-auto mt-6 grid max-w-md grid-cols-3 gap-px overflow-hidden rounded border border-line bg-line text-left">
          {[
            ["01", "Search", "Public web index"],
            ["02", "Crawl", "Company homepages"],
            ["03", "Analyse", "Signals & outreach"],
          ].map(([n, title, sub], i) => (
            <div key={n} className="bg-panel-2 px-3 py-3">
              <div className="font-mono text-[10px] text-brass">{n}</div>
              <div className="mt-1 text-[12px] text-ink">{title}</div>
              <div className="text-[11px] text-faint">{sub}</div>
              {running && (
                <div className="relative mt-2 h-px overflow-hidden bg-line-strong">
                  <span
                    className="animate-sheen absolute inset-y-0 w-1/2 bg-brass"
                    style={{ animationDelay: `${i * 200}ms` }}
                  />
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
