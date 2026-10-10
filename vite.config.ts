import { defineConfig, loadEnv } from "vite";

// In development the page calls /decide/*, which Vite forwards to the decision
// endpoint. Same-origin requests avoid CORS setup for any port or host.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const target = env.VITE_DECISION_BASE_URL || "http://localhost:11434";
  // A hosted endpoint's key is added here, in the dev server, so it never reaches the page.
  // It comes from the shell, or else from the gitignored .env.
  const apiKey = process.env.TYPESAFE_API_KEY || env.TYPESAFE_API_KEY;
  // Same for OpenAI's Decisions API: the page calls /openai/v1/decisions and the key is added here.
  const openaiKey = process.env.OPENAI_API_KEY || env.OPENAI_API_KEY;
  // /decide carries the key only when it leads to a hosted endpoint, never to a local Ollama.
  const decideHosted = !!apiKey && !/^https?:\/\/(localhost|127\.0\.0\.1)(:|\/|$)/.test(target);
  return {
    // Relative asset paths, so the build also works when bundled inside the iOS app.
    base: "./",
    // Tells the page that /decide leads to a hosted endpoint, so requests leave out Ollama-only fields.
    define: { "import.meta.env.VITE_DECISION_HOSTED": JSON.stringify(decideHosted ? "1" : "") },
    server: {
      port: 5173,
      proxy: {
        "/decide": {
          target,
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/decide/, ""),
          ...(decideHosted ? { headers: { Authorization: `Bearer ${apiKey}` } } : {}),
        },
        // TypeSafe's hosted Jev, next to a local /decide: the page sends "jev-latest" here.
        "/jev": {
          target: process.env.JEV_BASE_URL || env.JEV_BASE_URL || "https://api.typesafe.ai",
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/jev/, ""),
          ...(apiKey ? { headers: { Authorization: `Bearer ${apiKey}` } } : {}),
        },
        "/openai": {
          target: "https://api.openai.com",
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/openai/, ""),
          ...(openaiKey ? { headers: { Authorization: `Bearer ${openaiKey}` } } : {}),
        },
      },
    },
    test: {
      include: ["tests/**/*.test.ts"],
    },
  };
});
