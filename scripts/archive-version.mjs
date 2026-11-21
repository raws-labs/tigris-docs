#!/usr/bin/env node
// Freeze the current docs as a released version:
//   node scripts/archive-version.mjs X.Y.Z --capabilities operator-capabilities-v1.json
// Copies each top-level docs tree to src/content/docs/X.Y.Z/ with versioned slugs,
// links and imports, writes src/content/versions/X.Y.Z.json, registers the version
// in astro.config.mjs and CapabilityMatrix.astro, and stores the release's
// capability artifact as the X.Y.Z copy. The unversioned copy is left as it is.
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const docsDir = path.join(root, "src/content/docs");
const configPath = path.join(root, "astro.config.mjs");
const matrixPath = path.join(root, "src/components/CapabilityMatrix.astro");
const dataDir = path.join(root, "src/data");
const semver = /^\d+\.\d+\.\d+$/;

function fail(msg) {
  console.error(`archive-version: ${msg}`);
  process.exit(1);
}

const args = process.argv.slice(2);
const capIdx = args.indexOf("--capabilities");
const capPath = capIdx >= 0 ? args[capIdx + 1] : undefined;
const version = args.find((a, i) => i !== capIdx && i !== capIdx + 1);
if (!version || !semver.test(version) || !capPath || args.length !== 3) {
  fail("usage: node scripts/archive-version.mjs X.Y.Z --capabilities <operator-capabilities-v1.json>");
}
if (!fs.existsSync(capPath)) fail(`${capPath} does not exist`);
const capText = fs.readFileSync(capPath, "utf8");
try {
  if (!Array.isArray(JSON.parse(capText).operators)) throw new Error();
} catch {
  fail(`${capPath} is not a capability artifact with an operators list`);
}

const versionDir = path.join(docsDir, version);
const versionJson = path.join(root, "src/content/versions", `${version}.json`);
const capCopy = path.join(dataDir, `operator_capabilities_v1.${version}.json`);
for (const p of [versionDir, versionJson, capCopy]) {
  if (fs.existsSync(p)) fail(`${path.relative(root, p)} already exists`);
}

const cmp = (a, b) => {
  const x = a.split(".").map(Number);
  const y = b.split(".").map(Number);
  return x[0] - y[0] || x[1] - y[1] || x[2] - y[2];
};

// Insert `line` into a newest-first list of lines whose version `re` captures.
function insertSorted(lines, re, line) {
  const idx = lines.map((l, i) => [l.match(re), i]).filter(([m]) => m);
  if (idx.length === 0) fail(`no entry matching ${re} to place ${line.trim()} next to`);
  if (idx.some(([m]) => m[1] === version)) fail(`${version} is already listed: ${line.trim()}`);
  const before = idx.find(([m]) => cmp(m[1], version) < 0);
  lines.splice(before ? before[1] : idx.at(-1)[1] + 1, 0, line);
}

// Text of the bracketed literal that opens at `start`, skipping strings and comments.
function bracketed(src, start) {
  let depth = 0;
  for (let i = start; i < src.length; i++) {
    const c = src[i];
    if (c === '"' || c === "'" || c === "`") {
      for (i++; src[i] !== c; i++) if (src[i] === "\\") i++;
    } else if (c === "/" && src[i + 1] === "/") {
      i = src.indexOf("\n", i);
    } else if (c === "[" || c === "{") {
      depth++;
    } else if (c === "]" || c === "}") {
      if (--depth === 0) return src.slice(start, i + 1);
    }
  }
  fail("unbalanced brackets in astro.config.mjs");
}

// astro.config.mjs: sidebar for the version JSON, link-validator exclude, versions list.
let config = fs.readFileSync(configPath, "utf8");
const sidebarAt = config.search(/^\s*sidebar:\s*\[/m);
if (sidebarAt < 0) fail("no sidebar in astro.config.mjs");
const sidebar = vm.runInNewContext(`(${bracketed(config, config.indexOf("[", sidebarAt))})`);
const excludeRe = /(exclude:\s*\[[^\]]*?)(\s*\])/;
if (!excludeRe.test(config)) fail("no link-validator exclude list in astro.config.mjs");
config = config.replace(excludeRe, `$1, "/${version}/**"$2`);
const configLines = config.split("\n");
const entryRe = /^\s*\{ slug: "(\d+\.\d+\.\d+)", label: "v\1" \},$/;
const indent = configLines.find((l) => entryRe.test(l))?.match(/^\s*/)[0] ?? "";
insertSorted(configLines, entryRe, `${indent}{ slug: "${version}", label: "v${version}" },`);
config = configLines.join("\n");

// CapabilityMatrix.astro: import the archived artifact and map the slug to it.
const id = `v${version.replaceAll(".", "")}`;
const matrixLines = fs.readFileSync(matrixPath, "utf8").split("\n");
insertSorted(
  matrixLines,
  /^import v\d+ from "\.\.\/data\/operator_capabilities_v1\.(\d+\.\d+\.\d+)\.json";$/,
  `import ${id} from "../data/operator_capabilities_v1.${version}.json";`,
);
insertSorted(
  matrixLines,
  /^\s*"(\d+\.\d+\.\d+)": v\d+ as typeof develop,$/,
  `  "${version}": ${id} as typeof develop,`,
);

// Docs trees: every top-level directory that is not an archived version.
const trees = fs
  .readdirSync(docsDir, { withFileTypes: true })
  .filter((e) => e.isDirectory() && !semver.test(e.name))
  .map((e) => e.name);
const treeAlt = trees.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|");
const linkRe = new RegExp(`(\\]\\(|href=["']|^\\s*\\[[^\\]]+\\]:\\s*)/(${treeAlt})(?=[/#?)"'\\s]|$)`, "g");
// Single-line and side-effect imports, re-exports, and the last line of a multi-line import.
const importRe = /^(\s*(?:(?:import|export)\b[^"']*?\bfrom|import|\}\s*from)\s*["'])(\.{1,2}\/[^"']*)(["'];?\s*)$/;

function transform(rel, text) {
  if (!text.startsWith("---\n")) fail(`${rel} has no frontmatter`);
  const end = text.indexOf("\n---\n", 3);
  if (end < 0) fail(`${rel} has an unterminated frontmatter`);
  let front = text.slice(4, end);
  const own = front.match(/^slug:\s*["']?\/?(.*?)\/?["']?\s*$/m);
  if (own) {
    front = front.replace(own[0], `slug: ${version}/${own[1]}`);
  } else {
    const slug = rel.replace(/\.mdx?$/, "").replace(/(^|\/)index$/, "");
    front += `\nslug: ${version}/${slug}`;
  }
  const srcDir = path.dirname(path.join(docsDir, rel));
  const dstDir = path.dirname(path.join(versionDir, rel));
  let fence = null;
  const body = text
    .slice(end + 5)
    .split("\n")
    .map((line) => {
      const f = line.match(/^\s*(`{3,}|~{3,})/);
      if (f && (!fence || (f[1][0] === fence[0] && f[1].length >= fence.length))) {
        fence = fence ? null : f[1];
        return line;
      }
      if (fence) return line;
      line = line.replace(linkRe, `$1/${version}/$2`);
      if (!rel.endsWith(".mdx")) return line;
      return line.replace(importRe, (m, head, spec, tail) => {
        let target = path.resolve(srcDir, spec);
        const inDocs = path.relative(docsDir, target);
        if (trees.includes(inDocs.split(path.sep)[0])) target = path.join(versionDir, inDocs);
        let next = path.relative(dstDir, target).split(path.sep).join("/");
        if (!next.startsWith(".")) next = `./${next}`;
        return `${head}${next}${tail}`;
      });
    })
    .join("\n");
  return `---\n${front}\n---\n${body}`;
}

const pages = [];
const walk = (rel) => {
  for (const e of fs.readdirSync(path.join(docsDir, rel), { withFileTypes: true })) {
    const r = path.posix.join(rel, e.name);
    if (e.isDirectory()) walk(r);
    else pages.push(r);
  }
};
trees.forEach(walk);
const out = pages.map((rel) => {
  const src = path.join(docsDir, rel);
  const data = /\.mdx?$/.test(rel) ? transform(rel, fs.readFileSync(src, "utf8")) : fs.readFileSync(src);
  return [path.join(versionDir, rel), data];
});

// Everything is computed; write.
for (const [dst, data] of out) {
  fs.mkdirSync(path.dirname(dst), { recursive: true });
  fs.writeFileSync(dst, data);
}
fs.writeFileSync(versionJson, JSON.stringify({ sidebar, excluded: [] }, null, 2));
fs.writeFileSync(configPath, config);
fs.writeFileSync(matrixPath, matrixLines.join("\n"));
fs.writeFileSync(capCopy, capText);
console.log(`archived ${version}: ${out.length} files from ${trees.join(", ")}`);
