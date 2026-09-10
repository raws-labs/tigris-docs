import { defineCollection, z } from "astro:content";
import { glob } from "astro/loaders";
import { docsLoader } from "@astrojs/starlight/loaders";
import { docsSchema } from "@astrojs/starlight/schema";

// Native-Astro-blog approach: the blog is its OWN collection (not the docs
// collection), so it sits outside Starlight's routing/versioning middleware.
export const collections = {
  docs: defineCollection({ loader: docsLoader(), schema: docsSchema() }),
  blog: defineCollection({
    // Each post is a folder holding its index.md + co-located images. Strip the
    // trailing "/index" so `foo/index.md` still routes to /blog/foo/.
    loader: glob({
      pattern: "**/*.md",
      base: "./src/content/blog",
      generateId: ({ entry }) => entry.replace(/\.md$/, "").replace(/\/index$/, ""),
    }),
    schema: z.object({
      title: z.string(),
      date: z.coerce.date(),
      excerpt: z.string().optional(),
    }),
  }),
};
