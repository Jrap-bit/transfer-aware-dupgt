export type Pick = { item: number; title: string; stars: number };
export type Persona = { name: string; note?: string; picks: Pick[] };
export type Book = { item: number; title: string; category: string; n_ratings: number };
export type Top = {
  title: string; category: string; predicted: number; movie_avg: number; n: number; boost: number;
  alpha_high?: number; trust?: number; because?: { title: string; similarity: number } | null;
};
export type Column = {
  model: string; about: string; top: Top[];
  summary: { mean_alpha_high: number | null; mean_trust: number | null; mean_shift: number; mean_predicted: number; spread: number };
};
export type ProbeRow = { title: string; movie_avg: number; n: number; predicted: Record<string, { stars: number; alpha_high: number | null; trust: number | null }> };
export type Fallback = {
  title: string; movie_avg: number;
  bridges?: number[]; mu?: number; d?: number; w?: number; final?: number; w_const?: number; final_const?: number;
  probabilistic?: { mu: number; w: number; final: number };
};
export type Recommendation = {
  columns: Column[]; probe: ProbeRow[]; agreement: Record<string, number>; missing: string[];
  n_liked: number; n_disliked: number; n_candidates: number;
  fallback: Fallback[]; blend_params: { s0: number; tau2: number; c: number; tau2_const: number } | null;
  overlap: string; overlaps: string[];
};
export type MovieHit = { item: number; title: string; category: string; n_ratings: number; movie_avg: number };
export type RatedMovie = {
  item: number; title: string; movie_avg: number; n: number; predicted: Record<string, number>;
  alpha?: Record<string, number>; prob_w?: number; bridges?: number[]; d?: number; w?: number;
};
export type Rating = { movies: RatedMovie[]; models: string[]; overlap: string };
export type Status = { overlaps: string[]; missing: string[]; llm: { server: boolean; model_loaded: boolean } };
export type LlmPair = { pair_id: string; a: string; b: string; llm: string; models: Record<string, string> };
export type LlmResult =
  | { ok: true; seconds: number; prompt_tokens: number; completion_tokens: number; pairs: LlmPair[]; raw_json: unknown; prompt_preview: string }
  | { ok: false; error: string };
export type ReplayView = { pred: Record<string, number[]>; alpha_high: number[]; trust_w: (number | null)[]; mae: Record<string, number> };
export type ReplayUser = {
  user: string; n_books: number; books: { title: string; stars: number }[];
  movies: { title: string; actual: number }[]; by_overlap: Record<string, ReplayView>;
};
export type Replay = {
  overlaps: Record<string, string[]>; overall: Record<string, Record<string, { rmse: number; mae: number }>>;
  users: ReplayUser[]; shared_users: Record<string, number>;
};

async function call<T>(path: string, body?: unknown): Promise<T> {
  const r = await fetch(path, body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : undefined);
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j as T;
}

export const api = {
  status: () => call<Status>("/api/status"),
  personas: () => call<Persona[]>("/api/personas"),
  search: (q: string) => call<Book[]>("/api/search?q=" + encodeURIComponent(q)),
  searchMovies: (q: string) => call<MovieHit[]>("/api/movies?q=" + encodeURIComponent(q)),
  topical: (picks: Pick[], exclude: number[] = [], n = 8) => call<(MovieHit & { similarity: number })[]>("/api/topical", { picks, exclude, n }),
  rate: (picks: Pick[], items: number[], overlap = "10%") => call<Rating>("/api/rate", { picks, items, overlap }),
  suggest: (n = 8) => call<Book[]>("/api/suggest?n=" + n),
  recommend: (picks: Pick[], rank: string, overlap = "10%") => call<Recommendation>("/api/recommend", { picks, rank, overlap }),
  llm: (picks: Pick[], overlap = "10%") => call<LlmResult>("/api/llm", { picks, overlap }),
  replay: () => call<Replay>("/api/replay"),
};

export const MODEL_COLORS: Record<string, string> = {
  "Movie average": "#a3a3a3",
  "DUPGT": "#38bdf8",
  "+ Semantic gate": "#a78bfa",
  "+ LLM teacher": "#f472b6",
  "+ Shuffled teacher": "#fb7185",
  "Probabilistic bridge": "#34d399",
  "Uncertainty blend": "#facc15",
  "Ensemble mean": "#22d3ee",
  "Constant blend": "#fb923c",
};
export const SHORT: Record<string, string> = {
  "Movie average": "Movie avg",
  "DUPGT": "DUPGT",
  "+ Semantic gate": "Semantic",
  "+ LLM teacher": "LLM teacher",
  "+ Shuffled teacher": "Shuffled",
  "Probabilistic bridge": "Probabilistic",
  "Uncertainty blend": "Unc. blend",
  "Ensemble mean": "Ensemble",
  "Constant blend": "Const. blend",
};
export const SHARED: Record<string, string> = { "1%": "890", "10%": "8,903", "100%": "89,031" };
