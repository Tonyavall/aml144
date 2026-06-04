# Diagrams

D2 (https://d2lang.com) source for the architecture diagrams. Each `.d2` file is plain text
and is committed alongside its rendered `.svg` so the diagrams display on GitHub and in the
report without anyone needing the renderer installed.

| File | Shows |
| --- | --- |
| `modules.d2` | module dependency graph ("imports from", the layering) |
| `pipeline1.d2` | single-backbone frozen probe: train + predict data flow |
| `pipeline2.d2` | multi-view DINOv3 (deployed) with leak-free grouped CV |
| `pipeline3.d2` | cross-backbone ensemble fan-in and the weight-tuning gate |
| `probe.d2` | the cosine logistic probe and its cross-validation internals |

## Re-rendering

Install the `d2` CLI (https://github.com/terrastruct/d2, free and open source), then render a
file to SVG:

```bash
d2 docs/diagrams/pipeline2.d2 docs/diagrams/pipeline2.svg
```

Render the whole set:

```bash
for f in modules pipeline1 pipeline2 pipeline3 probe; do
  d2 docs/diagrams/$f.d2 docs/diagrams/$f.svg
done
```

Notes:

- `modules.d2` pins the `elk` layout engine inside the file (a `d2-config` var), so it renders
  the same way regardless of the `--layout` flag. The other files use the default engine.
- `d2 -w docs/diagrams/pipeline2.d2 out.svg` opens a live-reloading preview in the browser
  while editing.
- All `.d2` source is plain ASCII per the repo docs convention (connections use `->`).
