# Installation

## Requirements

- Python >= 3.10

## Install from PyPI

```bash
pip install sign-language-tools
```

This installs the core dependencies: `matplotlib`, `mediapipe`, `numpy`, `opencv-python`, `pandas`,
`plotly`, `scipy`, `scikit-learn`, `torch`, `tqdm` and `vidgear`.

!!! note
    The [`VideoPlayer`](player.md)'s random-access video backend prefers [PyAV](https://pyav.org/)
    or [decord](https://github.com/dmlc/decord) when available, falling back to `opencv-python`
    otherwise. Install one of them for faster seeking on long videos:

    ```bash
    pip install "sign-language-tools[video-backends]"
    ```

## Install for development

Clone the repository and install it in editable mode with the `dev` extra, which adds testing and
documentation tooling (`pytest`, `mkdocs`, `mkdocs-material`, `mkdocstrings`):

```bash
git clone https://github.com/ppoitier/sign-language-tools.git
cd sign-language-tools
pip install -e ".[dev]"
```

Serve the documentation locally with:

```bash
mkdocs serve
```

## Pairing with sign-language-data-loading

Most examples in this documentation load sample data with
[`sign-language-data-loading`](https://github.com/ppoitier/sign-language-data-loading), a separate
package for loading sign language datasets stored as WebDataset shards:

```bash
pip install sign-language-data-loading
```

See [Working with sign-language-data-loading](data-loading.md) for details.
