import { fileURLToPath } from 'node:url'

const repo = 'Semantic-Question-Answering-over-Large-Documents-using-RAG-LLaMA-3-Ollama'

// Pages serves the site from /<repo>/, local dev serves it from /.
// BASE_PATH can override (e.g. '' when deploying to a custom domain).
const basePath = process.env.BASE_PATH ?? (process.env.NODE_ENV === 'production' ? `/${repo}` : '')

/** @type {import('next').NextConfig} */
export default {
  output: 'export',              // static export — no server runtime
  images: { unoptimized: true }, // required: the Image Optimization API needs a server
  basePath,
  assetPrefix: basePath ? `${basePath}/` : '',
  trailingSlash: true,           // makes Pages resolve every route as a directory index
  // The repo root carries a stray, empty package-lock.json one level up from
  // here, which otherwise makes Next infer that directory as the workspace
  // root and breaks the `@/*` path alias during build.
  outputFileTracingRoot: fileURLToPath(new URL('.', import.meta.url)),
}
