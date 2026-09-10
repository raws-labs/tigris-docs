# tigris-docs

Documentation site for [TiGrIS](https://tigris-ml.dev), built with
[Astro Starlight](https://starlight.astro.build). Versioned by `tigris-ml`
release and deployed to tigris-ml.dev.

## Develop

```bash
npm install
npm run dev      # local dev server
npm run build    # static build to dist/
```

Docs live under `src/content/docs/` (the current "latest" version). Older
versions are frozen snapshots managed by
[`starlight-versions`](https://starlight-versions.vercel.app/); the blog is a
native Astro content collection under `src/content/blog/` + `src/pages/blog/`.
