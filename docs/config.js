// Summary Showdown site configuration. No build step: edit and push.
window.PAIRSORT_CONFIG = {
  repo: "ericflo/pairsort",
  // Default ballot path: a prefilled GitHub issue (zero backend). To collect ballots with your own backend
  // (e.g. a Cloudflare Worker or Supabase function), set submit_endpoint to a URL that accepts
  // POST application/json {ballot}. When set, the site POSTs there and shows the issue link as a fallback.
  submit_endpoint: null,
  votes_per_round: 8,
  // dimensions visitors are asked about (all six are ranked by the judges)
  human_dimensions: ["understandability", "writing", "verbosity", "completeness", "accuracy"],
};
