// Per-page Markdown endpoint: serves each docs page's raw Markdown at
// `<path>.md` (e.g. /architecture/tiling.md). Powers the "Copy for AI" /
// "View as Markdown" actions; complements the site-level /llms-*.txt endpoints.
import type { APIRoute, GetStaticPaths } from "astro";
import { getCollection } from "astro:content";

export const getStaticPaths: GetStaticPaths = async () => {
  const docs = await getCollection("docs");
  return docs.map((doc) => ({ params: { slug: doc.id }, props: { doc } }));
};

export const GET: APIRoute = async ({ props }) => {
  const doc = (props as { doc: any }).doc;
  const title: string = doc?.data?.title ?? "";
  const body: string = doc?.body ?? "";
  const markdown = title ? `# ${title}\n\n${body}` : body;
  return new Response(markdown, {
    headers: { "Content-Type": "text/markdown; charset=utf-8" },
  });
};
