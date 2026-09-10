// @ts-check
import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";
import starlightLlmsTxt from "starlight-llms-txt";
import starlightLinksValidator from "starlight-links-validator";
import starlightImageZoom from "starlight-image-zoom";
import starlightVersions from "starlight-versions";
// Versioning: re-enable at the FIRST frozen version (the first tigris-ml release
// we keep docs for). The spike proved starlight-versions works; it just rejects
// an empty version list, so it stays off until there is a version to freeze:
//   import starlightVersions from "starlight-versions";
//   ...plugins: [starlightVersions({ versions: [{ slug: "1.0" }] })]

// https://astro.build/config
export default defineConfig({
  site: "https://tigris-ml.dev",
  // Code highlighting for the native-Astro blog (Starlight's own pages use
  // Expressive Code instead). Light colors apply inline; the dark theme's token
  // colors ride along as --shiki-dark vars, toggled in BlogLayout's CSS.
  markdown: {
    shikiConfig: {
      themes: { light: "github-light", dark: "github-dark" },
    },
  },
  integrations: [
    starlight({
      title: "TiGrIS",
      description:
        "Tiled Graph Inference Scheduler — run ML models on embedded devices.",
      // "Edit page" link at the bottom of each doc → the docs repo.
      editLink: {
        baseUrl: "https://github.com/raws-labs/tigris-docs/edit/main/",
      },
      // AI-native endpoints: /llms.txt (index) and /llms-full.txt (whole site as
      // one document), plus per-page Markdown — the same pattern the Kinde docs
      // ship. Powers the "Copy for AI" / "View as Markdown" actions.
      plugins: [
        starlightLlmsTxt({
          projectName: "TiGrIS",
          description:
            "Tiled Graph Inference Scheduler — an ahead-of-time compiler and C99 runtime for running ML models on memory-constrained embedded devices.",
        }),
        // Click-to-zoom for images (architecture/tiling diagrams).
        starlightImageZoom(),
        // Build-time check for broken internal links and heading anchors.
        // The blog is a custom (non-docs) collection the validator can't model;
        // the frozen 0.6.0 snapshot is carved down (no tutorials) so isn't
        // re-validated. Both are excluded.
        starlightLinksValidator({ exclude: ["/blog/**", "/0.6.0/**"] }),
        // Versioned docs. Current top-level docs = the forward-looking line
        // ("0.7.0"); 0.6.0 is a frozen snapshot (tutorials carved out — they
        // are post-0.6.0). Because we override ThemeSelect/PageTitle, the plugin
        // steps aside and we render VersionSelect/VersionNotice ourselves.
        starlightVersions({
          // Current top-level docs = the in-development line. "latest" is generic
          // (the next release could be v0.6.1 or v0.7.0); 0.6.0 is the frozen
          // release. URL slugs stay clean (/0.6.0/); labels carry the "v".
          current: { label: "latest" },
          versions: [{ slug: "0.6.0", label: "v0.6.0" }],
        }),
      ],
      // "Last updated" from git commit history, at the bottom of each doc.
      lastUpdated: true,
      // Starlight emits the SVG favicon by default; add the .ico fallback (older
      // browsers) and the apple-touch icon so docs pages carry the same favicon
      // set as the standalone homepage.
      head: [
        { tag: "link", attrs: { rel: "icon", href: "/favicon.ico", sizes: "32x32" } },
        { tag: "link", attrs: { rel: "apple-touch-icon", href: "/apple-touch-icon.png", sizes: "180x180" } },
      ],
      // Code syntax theme. `themes` is [darkTheme, lightTheme]; swap either for
      // any bundled Shiki theme name (full list: https://shiki.style/themes).
      // Muted picks that fit here: "vitesse-dark"/"vitesse-light" (current),
      // "github-dark"/"github-light", "min-dark"/"min-light",
      // "one-dark-pro"/"one-light", "rose-pine"/"rose-pine-dawn",
      // "catppuccin-mocha"/"catppuccin-latte", "night-owl"/"github-light".
      // Style overrides below strip the editor-frame chrome for a flat panel.
      expressiveCode: {
        themes: ["github-dark", "github-light"],
        styleOverrides: {
          borderRadius: "0",
          borderColor: "transparent",
          codeBackground: "var(--tigris-code-bg)",
          codePaddingInline: "1.25rem",
          codePaddingBlock: "1rem",
          frames: {
            frameBoxShadowCssValue: "none",
            // Terminal/editor frames use their own backgrounds; point both at the
            // soft-gray panel so every block matches (the header is hidden in CSS).
            editorBackground: "var(--tigris-code-bg)",
            terminalBackground: "var(--tigris-code-bg)",
          },
        },
      },
      customCss: [
        "@fontsource-variable/inter/index.css",
        "@fontsource-variable/jetbrains-mono/index.css",
        "./src/styles/brand.css",
      ],
      components: {
        Footer: "./src/components/Footer.astro",
        PageTitle: "./src/components/PageTitle.astro",
        ThemeSelect: "./src/components/ThemeSelect.astro",
        Header: "./src/components/Header.astro",
        // "Edit this page" moves from the page footer to the top of the right
        // TOC (Kinde-style); EditLink is emptied so it no longer renders below.
        EditLink: "./src/components/EditLink.astro",
        TableOfContents: "./src/components/TableOfContents.astro",
        // Empty: "Last updated" moves into the TOC header (TableOfContents).
        LastUpdated: "./src/components/LastUpdated.astro",
        // On narrow screens the rail is hidden; overlay the tools on the mobile
        // "On this page" bar so they don't vanish.
        MobileTableOfContents: "./src/components/MobileTableOfContents.astro",
        // Empty: the header already has GitHub + theme; drop the duplicate at the
        // bottom of the mobile menu.
        MobileMenuFooter: "./src/components/MobileMenuFooter.astro",
      },
      social: [
        { icon: "github", label: "GitHub", href: "https://github.com/raws-labs/tigris" },
      ],
      // Chapters fold (Kinde-style); Starlight auto-expands the group holding
      // the current page.
      sidebar: [
        { label: "Getting Started", collapsed: true, items: [{ autogenerate: { directory: "getting-started" } }] },
        { label: "CLI Reference", collapsed: true, items: [{ autogenerate: { directory: "toolchain" } }] },
        { label: "Architecture", collapsed: true, items: [{ autogenerate: { directory: "architecture" } }] },
        { label: "Runtime", collapsed: true, items: [{ autogenerate: { directory: "runtime" } }] },
        { label: "Tutorials", collapsed: true, items: [{ autogenerate: { directory: "tutorials" } }] },
      ],
    }),
  ],
});
