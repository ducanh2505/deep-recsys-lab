import { access, readFile, readdir, stat } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { join } from "node:path";

const outputRoot = fileURLToPath(new URL("../dist/", import.meta.url));
const forbidden = [
  ["Movie", "Lens"].join(""),
  ["Ye", "lp"].join(""),
  ["bench", "mark"].join(""),
  ["re", "production"].join(""),
  ["og", ":", "image"].join(""),
  ["__SITE", "_ORIGIN__"].join(""),
];

const repositoryRoot = fileURLToPath(new URL("../../../", import.meta.url));
const retiredHostingFile = join(
  repositoryRoot,
  "." + ["open", "ai"].join(""),
  ["hosting", ".json"].join(""),
);
try {
  await access(retiredHostingFile);
  throw new Error("site must remain standalone");
} catch (error) {
  if (error?.code !== "ENOENT") throw error;
}

async function filesUnder(path) {
  const entries = await readdir(path);
  const files = [];
  for (const entry of entries) {
    const child = join(path, entry);
    if ((await stat(child)).isDirectory()) files.push(...(await filesUnder(child)));
    else files.push(child);
  }
  return files;
}

const files = await filesUnder(outputRoot);
if (!files.some((file) => file.endsWith("index.html"))) {
  throw new Error("site build is missing index.html");
}

for (const file of files.filter((path) => /\.(html|css|js)$/.test(path))) {
  const content = await readFile(file, "utf8");
  for (const token of forbidden) {
    if (content.includes(token)) throw new Error(`${file} contains forbidden token: ${token}`);
  }
  if (/\b(?:href|src)=["']\/(?!\/)/.test(content)) {
    throw new Error(`${file} contains a root-absolute asset or link`);
  }
}

console.log(`validated ${files.length} portable site assets`);
