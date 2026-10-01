# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "altair==6.3.0",
#     "marimo>=0.25.0",
#     "numpy==2.5.3",
#     "polars==1.44.2",
#     "remotezip==0.12.6",
#     "scikit-learn==1.9.1",
#     "torch==2.11.0",
#     "transformers==5.18.0",
#     "vegafusion==2.0.3",
#     "vl-convert-python==1.9.0.post1",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def title(mo):
    mo.md(r"""
    # Unit 7 assignment

    In the last assignment you explored one deep mutational scanning (DMS) dataset from [ProteinGym](https://proteingym.org). In this one you will build a machine learning model that predicts `DMS_score` from the protein sequence.

    You will:

    1. Load **ESMC**, a protein language model, and tokenize the wild-type sequence.
    2. Turn every mutant sequence into a vector of numbers (an **embedding**).
    3. Look at the embeddings with PCA and t-SNE.
    4. Train a random forest on the embeddings and test how well it predicts `DMS_score`.
    """)
    return


@app.cell
def imports_mo():
    import marimo as mo

    return (mo,)


@app.cell
def imports():
    from collections import Counter

    import altair as alt
    import numpy as np
    import polars as pl
    import torch
    from remotezip import RemoteZip
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.manifold import TSNE
    from sklearn.model_selection import train_test_split
    from transformers import AutoModel, AutoTokenizer

    # Some ProteinGym datasets have far more than altair's default 5000-row limit.
    alt.data_transformers.enable("vegafusion")
    return Counter, PCA, RemoteZip, TSNE, np, pl, torch


@app.cell(hide_code=True)
def gpu_note(mo):
    mo.md(r"""
    ## Check that the GPU is on

    This notebook runs a protein language model. It needs a GPU.

    In molab, click the **notebook specs** button in the header (top right) and turn the GPU on. molab restarts the notebook when you do this. That is normal.

    The cell below checks for the GPU. Do not go on until it says the GPU is ready.
    """)
    return


@app.cell(hide_code=True)
def gpu_check(mo, torch):
    if torch.cuda.is_available():
        _gpu = torch.cuda.get_device_name(0)
        _vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        _msg = mo.callout(
            mo.md(f"**GPU is ready:** {_gpu} with {_vram:.0f} GB of memory."),
            kind="success",
        )
    else:
        _msg = mo.callout(
            mo.md(
                "**No GPU found.** Click the notebook specs button in the molab header and turn the GPU on. Then wait for the notebook to restart."
            ),
            kind="danger",
        )
    _msg
    return


@app.cell(hide_code=True)
def dataset_note(mo):
    mo.md(r"""
    ## Pick your dataset

    Replace the value of `DATASET_ID` below with the ProteinGym DMS ID you chose in the first notebook.

    One limit: ESMC can only read proteins up to 2,046 amino acids long. A few ProteinGym proteins are longer. If you picked one of those, the next cell will stop and tell you to pick a different dataset.
    """)
    return


@app.cell
def dataset_id():
    DATASET_ID = "BLAT_ECOLX_Deng_2012"
    return (DATASET_ID,)


@app.cell
def load_df(Counter, DATASET_ID, RemoteZip, mo, pl):
    # ESMC reads at most 2048 tokens (2046 amino acids). These four ProteinGym proteins are longer.
    TOO_LONG_FOR_ESMC = {
        "A0A140D2T1_ZIKV_Sourisseau_2019",
        "BRCA2_HUMAN_Erwood_2022_HEK293T",
        "POLG_HCVJF_Qi_2014",
        "POLG_CXB3N_Mattenberger_2021",
    }
    mo.stop(
        DATASET_ID in TOO_LONG_FOR_ESMC,
        mo.callout(
            mo.md(
                f"**`{DATASET_ID}` is too long for ESMC.** ESMC can read at most 2,046 amino acids. "
                "Go back and pick a different dataset."
            ),
            kind="danger",
        ),
    )

    PROTEINGYM_SUBSTITUTIONS_URL = (
        "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/"
        "DMS_ProteinGym_substitutions.zip"
    )

    with RemoteZip(PROTEINGYM_SUBSTITUTIONS_URL) as _zf:
        _match = next(n for n in _zf.namelist() if DATASET_ID in n)
        with _zf.open(_match) as _f:
            df = pl.read_csv(_f)

    # The wild-type (WT) sequence is not in the table. Each mutant differs from WT at
    # only one or a few positions, so the most common letter at each position is WT.
    WT_SEQUENCE = "".join(
        Counter(_column).most_common(1)[0][0]
        for _column in zip(*df["mutated_sequence"].to_list())
    )

    df
    return WT_SEQUENCE, df


@app.cell(hide_code=True)
def wt_show(WT_SEQUENCE, df, mo):
    mo.md(f"""
    The table has **{len(df):,} mutants**. The wild-type sequence has **{len(WT_SEQUENCE)} amino acids**:

    `{WT_SEQUENCE}`
    """)
    return


@app.cell(hide_code=True)
def ex1_md(mo):
    mo.md(r"""
    ## Exercise 1: load ESMC and tokenize the wild-type sequence

    [ESMC](https://huggingface.co/biohub/ESMC-600M) is a protein language model. It was trained on billions of protein sequences. It reads a sequence one amino acid at a time and turns each amino acid into a vector of numbers. We use the 600 million parameter version, `biohub/ESMC-600M`, through the Hugging Face `transformers` library.

    A **tokenizer** turns the letters of a sequence into integer IDs the model understands.

    1. Load the tokenizer with `AutoTokenizer.from_pretrained(...)`. Call it `tokenizer`.
    2. Load the model with `AutoModel.from_pretrained(...)`. Pass `dtype=torch.bfloat16` so it uses less memory. Move it to the GPU with `.to("cuda")` and switch off training mode with `.eval()`. Call it `model`.
    3. Tokenize `WT_SEQUENCE` with `tokenizer(WT_SEQUENCE, return_tensors="pt")`. Call the result `wt_tokens` and show it.

    Look at the shape of `wt_tokens["input_ids"]`. It is `(1, number_of_tokens)`. The `1` is because you tokenized one sequence.

    <details>
    <summary>Hint</summary>

    ```python
    tokenizer = AutoTokenizer.from_pretrained("biohub/ESMC-600M")
    model = AutoModel.from_pretrained("biohub/ESMC-600M", dtype=torch.bfloat16).to("cuda").eval()

    wt_tokens = tokenizer(WT_SEQUENCE, return_tensors="pt")
    wt_tokens
    ```

    </details>
    """)
    return


@app.cell
def ex1():
    # Code goes here
    return


@app.cell(hide_code=True)
def ex1_assert_task(mo):
    mo.md(r"""
    Now write a sanity check with `assert`. The tokenizer adds a `<cls>` token at the start and an `<eos>` token at the end. So the number of tokens should be the number of amino acids in `WT_SEQUENCE` plus 2.

    Check that `wt_tokens["input_ids"]` has shape `(1, len(WT_SEQUENCE) + 2)`.

    <details>
    <summary>Hint</summary>

    ```python
    assert wt_tokens["input_ids"].shape == (1, len(WT_SEQUENCE) + 2)
    ```

    </details>
    """)
    return


@app.cell
def ex1_assert():
    # Code goes here
    return


@app.cell(hide_code=True)
def ex2_md(mo):
    mo.md(r"""
    ## Exercise 2: embed every mutant sequence

    When the model reads a sequence it outputs one vector per token. This output is called the **last hidden state**. Its shape is `(batch, tokens, hidden_size)`. The `hidden_size` is a fixed property of the model. You can read it from `model.config.hidden_size`.

    We want **one** vector per sequence, not one per token. The function below averages the token vectors over the real amino acids (it skips the `<cls>`, `<eos>` and padding tokens). The result is a numpy array with one row per sequence and `hidden_size` columns.

    Run the two cells below to embed every sequence in `df`. It should take well under a minute on the GPU.
    """)
    return


@app.cell
def embed_sequences_fn(mo, model, np, tokenizer, torch):
    def embed_sequences(sequences, batch_size=64):
        """Return one embedding per sequence as a numpy array of shape (n_sequences, hidden_size)."""
        special_ids = torch.tensor(
            [
                tokenizer.cls_token_id,
                tokenizer.eos_token_id,
                tokenizer.pad_token_id,
            ],
            device=model.device,
        )
        chunks = []
        with torch.inference_mode():
            for start in mo.status.progress_bar(
                range(0, len(sequences), batch_size), title="Embedding"
            ):
                batch = tokenizer(
                    sequences[start : start + batch_size],
                    return_tensors="pt",
                    padding=True,
                )
                batch = {k: v.to(model.device) for k, v in batch.items()}
                hidden = model(
                    **batch
                ).last_hidden_state  # (batch, tokens, hidden_size)
                keep = ~torch.isin(
                    batch["input_ids"], special_ids
                )  # True only for real amino acids
                keep = keep.unsqueeze(-1).to(
                    hidden.dtype
                )  # (batch, tokens, 1)
                mean = (hidden * keep).sum(dim=1) / keep.sum(
                    dim=1
                )  # (batch, hidden_size)
                chunks.append(mean.float().cpu().numpy())
        return np.concatenate(chunks)

    return (embed_sequences,)


@app.cell
def embed_all(df, embed_sequences, mo, model, tokenizer):
    # This cell needs `tokenizer` and `model` from Exercise 1. It waits until they exist.
    try:
        tokenizer, model
    except NameError:
        mo.stop(
            True,
            mo.callout(
                mo.md(
                    "**Finish Exercise 1 first.** This cell needs `tokenizer` and `model`."
                ),
                kind="warn",
            ),
        )

    embeddings = embed_sequences(df["mutated_sequence"].to_list())
    embeddings.shape
    return (embeddings,)


@app.cell(hide_code=True)
def ex2_task(mo):
    mo.md(r"""
    Now write a sanity check. Use `assert` statements to confirm that:

    1. `embeddings` has one row per mutated protein sequence in `df`, and one column per hidden dimension (`model.config.hidden_size`).
    2. `embeddings` has no `NaN` values.

    If an `assert` fails Python raises an error. If the cell runs without an error, the checks passed.

    <details>
    <summary>Hint</summary>

    ```python
    assert embeddings.shape == (len(df), model.config.hidden_size)
    assert not np.isnan(embeddings).any()
    ```

    </details>
    """)
    return


@app.cell
def ex2():
    # Code goes here
    return


@app.cell(hide_code=True)
def ex3_md(mo):
    mo.md(r"""
    ## Exercise 3: look at the embeddings with PCA and t-SNE

    Each embedding has over a thousand dimensions. You cannot plot that. **PCA** and **t-SNE** squash the data down to two dimensions so you can. PCA keeps the directions with the most spread. t-SNE tries to keep points that are close in the full space close in the picture.

    The function below runs both and returns one long table. Each sequence shows up twice, once with `method = "PCA"` and once with `method = "t-SNE"`. The 2D coordinates are in the `x` and `y` columns. The table also carries `mutant` and `DMS_score`.
    """)
    return


@app.cell
def reduce_dimensions_fn(PCA, TSNE, df, embeddings, pl):
    def reduce_dimensions(embeddings, labels):
        """Project embeddings to 2D with PCA and t-SNE. Returns a long table with one row per (sequence, method)."""
        pca = PCA(n_components=50, random_state=0).fit_transform(embeddings)
        tsne = TSNE(n_components=2, random_state=0).fit_transform(
            pca
        )  # t-SNE on the top 50 PCs is faster
        frames = []
        for method, coords in [("PCA", pca[:, :2]), ("t-SNE", tsne)]:
            frames.append(
                labels.with_columns(
                    method=pl.lit(method),
                    x=pl.Series(coords[:, 0]),
                    y=pl.Series(coords[:, 1]),
                )
            )
        return pl.concat(frames)

    coords = reduce_dimensions(embeddings, df.select("mutant", "DMS_score"))
    coords
    return


@app.cell(hide_code=True)
def ex3_task(mo):
    mo.md(r"""
    Plot `coords` as two scatter plots side by side, one for PCA and one for t-SNE. Put `x` on the x axis and `y` on the y axis. Color each point by `DMS_score`. Use `column=alt.Column("method")` to get the two panels.

    By default Altair shares the axes between panels. PCA and t-SNE are on very different scales, so add `.resolve_scale(x="independent", y="independent")` at the end.

    Do mutants with high and low scores land in different parts of the picture?

    <details>
    <summary>Hint</summary>

    ```python
    alt.Chart(coords, width=350, height=350).mark_circle(size=20).encode(
        x=alt.X("x").title(None),
        y=alt.Y("y").title(None),
        color=alt.Color("DMS_score").scale(scheme="viridis"),
        column=alt.Column("method").title(None),
        tooltip=["mutant", "DMS_score"],
    ).resolve_scale(x="independent", y="independent")
    ```

    </details>
    """)
    return


@app.cell
def ex3():
    # Code goes here
    return


@app.cell(hide_code=True)
def ex4_md(mo):
    mo.md(r"""
    ## Exercise 4: predict DMS_score with a random forest

    Now train a model. The features `X` are the embeddings. The target `y` is `DMS_score`. The cell below sets them up.
    """)
    return


@app.cell
def xy_setup(df, embeddings):
    X = embeddings
    y = df["DMS_score"].to_numpy()
    X.shape, y.shape
    return


@app.cell(hide_code=True)
def ex4_task(mo):
    mo.md(r"""
    You need three things from scikit-learn. They are already imported at the top of the notebook.

    | Tool | What it does | Docs |
    | --- | --- | --- |
    | `train_test_split(X, y, test_size=..., random_state=...)` | Splits the data into a training part and a test part. Returns four arrays: `X_train, X_test, y_train, y_test`. | [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) |
    | `RandomForestRegressor(n_estimators=..., n_jobs=..., random_state=...)` | A model made of many decision trees. It averages their answers. | [RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) |
    | `.fit(X_train, y_train)` | Trains the model. Every scikit-learn model has this method. | [fit](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor.fit) |
    | `.predict(X_test)` | Uses the trained model to guess `y` for new data. | [predict](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor.predict) |

    Now do this:

    1. Split the data. Keep a random 20% aside for testing with `train_test_split`. Pass `random_state=0` so you get the same split each time.
    2. Make a `RandomForestRegressor` and train it on the training data with `.fit`. 100 trees is enough. Pass `n_jobs=-1` to use every CPU.
    3. Predict `DMS_score` for the test data with `.predict`.
    4. Put the actual and predicted test scores in a polars DataFrame called `results` with columns `actual` and `predicted`.
    5. Compute the **Spearman correlation** between them. ProteinGym uses Spearman because it only cares about ranking mutants in the right order. Polars can do it with `pl.corr("actual", "predicted", method="spearman")`.
    6. Plot `predicted` against `actual` as a scatter chart. Put the Spearman value in the title.

    <details>
    <summary>Hint</summary>

    ```python
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=0).fit(X_train, y_train)

    results = pl.DataFrame({"actual": y_test, "predicted": rf.predict(X_test)})
    spearman = results.select(pl.corr("actual", "predicted", method="spearman")).item()

    alt.Chart(results, width=400, height=400, title=f"Spearman = {spearman:.2f}").mark_circle().encode(
        x="actual",
        y="predicted",
    )
    ```

    </details>
    """)
    return


@app.cell
def ex4():
    # Code goes here
    return


if __name__ == "__main__":
    app.run()

