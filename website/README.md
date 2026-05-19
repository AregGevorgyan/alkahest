# Alkahest landing page

Organization homepage for [alkahest-cas.github.io](https://alkahest-cas.github.io/).

## Edit

- `src/index.html` — page content and links
- `src/styles.css` — styles

## Local preview

Requires [Bun](https://bun.sh):

```bash
cd website
bun run dev
```

## Build

Copies `src/*` to `website/` root (gitignored artifacts used for local preview only):

```bash
bun run build
```

## Deploy

Pushes to [alkahest-cas/alkahest-cas.github.io](https://github.com/alkahest-cas/alkahest-cas.github.io) on every push to `main` that touches `website/**` (see `.github/workflows/website.yml`). CI publishes `website/src/` to the Pages site root.

Full docs live at [alkahest-cas.github.io/alkahest/](https://alkahest-cas.github.io/alkahest/). The interactive playground is at [alkahest-cas.github.io/playground/](https://alkahest-cas.github.io/playground/).
