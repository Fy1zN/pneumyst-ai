"use client";

import { useEffect, useRef, useState, type FormEvent } from "react";

import { ApiError, type AskAnswer } from "@/lib/api";
import { useApi } from "@/lib/use-api";

// The same limits as the backend: AskRequest takes 3 to 500 characters, and RATE_ASK_PER_HOUR defaults to 20.
const MIN_LENGTH = 3;
const MAX_LENGTH = 500;

const SUGGESTIONS = [
  "Why was this finding flagged?",
  "What does the heatmap show?",
  "How reliable are these models?",
  "Compare with my last scan",
];

type Turn = { id: number; question: string; answer?: AskAnswer; error?: string };

function askError(error: unknown): string {
  if (!(error instanceof ApiError)) return "Something went wrong on our side. Try again.";
  if (error.code === "NETWORK_ERROR") return error.message;
  if (error.status === 401 || error.code === "AUTH_UNAVAILABLE") return "Your session has ended. Sign in again to ask questions.";
  if (error.status === 429) {
    const minutes = error.retryAfterSeconds ? Math.max(1, Math.ceil(error.retryAfterSeconds / 60)) : null;
    return `You have reached the limit of 20 questions an hour.${minutes ? ` Try again in ${minutes} ${minutes === 1 ? "minute" : "minutes"}.` : ""}`;
  }
  if (error.status === 422) return `Questions need ${MIN_LENGTH} to ${MAX_LENGTH} characters.`;
  if (error.code === "ASSISTANT_UNAVAILABLE") return "The assistant could not answer right now. Try again shortly.";
  return "Something went wrong on our side. Try again.";
}

function Answer({ answer }: { answer: AskAnswer }) {
  return (
    <div className="flex flex-col gap-3 rounded-xl border border-line bg-white p-4">
      <p className={`label text-[0.5625rem] font-bold ${answer.refused ? "text-band-possible" : "text-brand-green"}`}>
        {answer.refused ? "Outside what the assistant answers" : "Answer"}
      </p>
      <div className="flex flex-col gap-2 text-sm leading-relaxed">
        {answer.answer
          .split(/\n\s*\n/)
          .filter(Boolean)
          .map((paragraph, i) => (
            <p key={i}>{paragraph}</p>
          ))}
      </div>
      {answer.tools_used.length > 0 && (
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="label text-[0.5rem] text-muted">Checked</span>
          {[...new Set(answer.tools_used)].map((tool) => (
            <span key={tool} className="rounded-full border border-line bg-page px-2.5 py-1 font-mono text-[0.625rem] text-muted">
              {tool}
            </span>
          ))}
        </div>
      )}
      <p className="text-xs text-muted">{answer.disclaimer}</p>
    </div>
  );
}

/** Questions to the assistant about one scan. Nothing here is stored: the conversation is gone when the page closes. */
export function AskTab({ analysisId }: { analysisId: string }) {
  const api = useApi();
  const [turns, setTurns] = useState<Turn[]>([]);
  const [draft, setDraft] = useState("");
  const [pending, setPending] = useState(false);
  const nextId = useRef(0);
  const last = useRef<HTMLLIElement>(null);

  useEffect(() => {
    last.current?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [turns]);

  async function ask(question: string, retryId?: number) {
    const id = retryId ?? nextId.current++;
    setPending(true);
    setTurns((current) =>
      retryId === undefined ? [...current, { id, question }] : current.map((t) => (t.id === id ? { id, question } : t)),
    );
    try {
      const answer = await api<AskAnswer>(`/analyses/${analysisId}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      setTurns((current) => current.map((t) => (t.id === id ? { ...t, answer } : t)));
    } catch (error) {
      setTurns((current) => current.map((t) => (t.id === id ? { ...t, error: askError(error) } : t)));
    } finally {
      setPending(false);
    }
  }

  function submit(event: FormEvent) {
    event.preventDefault();
    const question = draft.trim();
    if (question.length < MIN_LENGTH || pending) return;
    setDraft("");
    void ask(question);
  }

  const tooShort = draft.trim().length < MIN_LENGTH;

  return (
    <div className="flex flex-1 flex-col gap-4">
      <div className="flex flex-col gap-2">
        <h2 className="font-serif text-2xl text-brand-green italic">Ask about this scan</h2>
        <p className="text-sm text-muted">
          Answers use only this scan, your earlier scans and the model facts. Case details and symptoms are never sent. Your questions are not
          saved.
        </p>
      </div>

      <div className="flex flex-wrap gap-2">
        {SUGGESTIONS.map((suggestion) => (
          <button
            key={suggestion}
            type="button"
            disabled={pending}
            onClick={() => void ask(suggestion)}
            className="rounded-full border border-brand-mint-strong bg-brand-mint px-3 py-1.5 text-left text-xs font-medium text-brand-green-deep transition hover:border-brand-green disabled:opacity-50"
          >
            {suggestion}
          </button>
        ))}
      </div>

      {turns.length > 0 && (
        <ol className="flex flex-col gap-3" aria-live="polite">
          {turns.map((turn, index) => (
            <li key={turn.id} ref={index === turns.length - 1 ? last : undefined} className="flex flex-col gap-3">
                           <p className="ml-8 self-end rounded-xl rounded-br-sm border border-brand-mint-strong bg-brand-mint px-4 py-2.5 text-sm text-brand-green-deep">
                {turn.question}
              </p>
              {turn.answer ? (
                <Answer answer={turn.answer} />
              ) : turn.error ? (
                <div className="flex flex-col items-start gap-2 rounded-xl bg-caution p-4 text-sm">
                  <p className="text-band-possible">{turn.error}</p>
                  <button
                    type="button"
                    disabled={pending}
                    onClick={() => void ask(turn.question, turn.id)}
                    className="font-semibold text-brand-green underline underline-offset-4 disabled:opacity-50"
                  >
                    Try again
                  </button>
                </div>
              ) : (
                <p className="flex items-center gap-2.5 rounded-xl border border-line bg-white p-4 text-sm text-muted">
                  <svg viewBox="0 0 20 20" className="size-4 animate-spin motion-reduce:animate-none" aria-hidden="true">
                    <circle cx="10" cy="10" r="8" fill="none" stroke="#bbf7d0" strokeWidth="2.5" />
                    <path d="M10 2a8 8 0 0 1 8 8" fill="none" stroke="#16a34a" strokeWidth="2.5" strokeLinecap="round" />
                  </svg>
                  Checking the scan and writing an answer…
                </p>
              )}
            </li>
          ))}
        </ol>
      )}

      <form onSubmit={submit} className="mt-auto flex flex-col gap-2 pt-2">
        <div className="flex items-center gap-2 rounded-full border border-line bg-white py-1.5 pr-1.5 pl-4 focus-within:border-brand-green">
          <label htmlFor="ask-question" className="sr-only">
            Ask a question about this scan
          </label>
          <input
            id="ask-question"
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            maxLength={MAX_LENGTH}
            placeholder="Ask a question about this scan"
            autoComplete="off"
            // The pill's border shows focus; the global focus outline would draw a second box inside it.
            className="min-w-0 flex-1 bg-transparent text-sm outline-none placeholder:text-muted focus-visible:outline-none!"
          />
          <button
            type="submit"
            disabled={pending || tooShort}
            className="glass-active rounded-full border px-5 py-2 text-sm font-semibold text-brand-green-deep transition disabled:opacity-50"
          >
            {pending ? "Sending…" : "Send"}
          </button>
        </div>
        <p className="flex justify-between px-1 text-xs text-muted">
          <span>
            {draft.length} / {MAX_LENGTH}
          </span>
          <span>Up to 20 questions an hour</span>
        </p>
      </form>
    </div>
  );
}