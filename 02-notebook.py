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

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def title(mo):
    mo.md(r"""
    # Unit 8 assignment

    In the last assignment you split your ProteinGym data set 80% training and 20% test examples at random. You trained a random forest on ESMC embeddings using the training data and got a Spearman correlation on a random 20% test set. When you split these protein variant data at random, you end up with training and test data covering the same positions. In this exercise we will investigate whether our models can predict protein activity when the test set has variants from different positions and entirely different contiguous regions than the training data.

    ProteinGym provides three kinds of splits for each data set:

    | Split | How the folds are made | What it tests |
    | --- | --- | --- |
    | **random** | Shuffle the mutants and deal them into 5 folds. | The easy case. Same positions in train and test. |
    | **modulo** | Sort positions. Every 5th position goes in the same fold. | New positions, but neighbours of training data are in the test set. |
    | **contiguous** | Cut the protein into 5 blocks of positions. | A whole new region of the protein. The hard case. |

    You will:

    1. Visualize the three splits.
    2. Write a function that evaluates models on each train/test split.
    3. Compare **ridge regression** and a **random forest** on all train/test splits.
    4. Plot the results and explain them.
    """)
    return


@app.cell
def imports_mo():
    import marimo as mo

    return (mo,)


@app.cell
def imports():
    import altair as alt
    import numpy as np
    import polars as pl
    import torch
    from remotezip import RemoteZip
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import RidgeCV
    from transformers import AutoModel, AutoTokenizer

    # Some ProteinGym datasets have far more than altair's default 5000-row limit.
    alt.data_transformers.enable("vegafusion")
    return AutoModel, AutoTokenizer, RemoteZip, np, pl, torch


@app.cell(hide_code=True)
def gpu_note(mo):
    mo.md(r"""
    ## Check that the GPU is on

    This notebook runs a protein language model. It needs a GPU to run efficiently.

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

    Replace the value of `DATASET_ID` below with the ProteinGym DMS ID you used in the last two notebooks.

    In this notebook we also load ProteinGym's **cross-validation file** for your dataset. It has the same columns as before plus three new ones. The cell below renames them to `random`, `modulo` and `contiguous`. Each one puts every mutant into a fold numbered 0 to 4.

    One limit: ESMC can only read proteins up to 2,046 amino acids long. A few ProteinGym proteins are longer. If you picked one of those, the next cell will stop and tell you to pick a different dataset.
    """)
    return


@app.cell
def dataset_id():
    DATASET_ID = "BLAT_ECOLX_Deng_2012"
    return (DATASET_ID,)


@app.cell
def load_df(DATASET_ID, RemoteZip, mo, pl):
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

    PROTEINGYM_CV_FOLDS_URL = (
        "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/"
        "cv_folds_singles_substitutions.zip"
    )

    with RemoteZip(PROTEINGYM_CV_FOLDS_URL) as _zf:
        _match = next(n for n in _zf.namelist() if DATASET_ID in n)
        with _zf.open(_match) as _f:
            df = pl.read_csv(_f)

    # Give the three fold columns short names.
    df = df.rename(
        {
            "fold_random_5": "random",
            "fold_modulo_5": "modulo",
            "fold_contiguous_5": "contiguous",
        }
    )

    df
    return (df,)


@app.cell(hide_code=True)
def ex1_md(mo):
    mo.md(r"""
    ## Exercise 1: picture the three splits

    The cell below reshapes the fold columns into one long table called `folds_long`. It has one row per mutant **per scheme**, with these columns:

    | Column | Meaning |
    | --- | --- |
    | `mutant` | The mutation ID, like `H24C`. |
    | `position` | The position number pulled out of `mutant`. |
    | `scheme` | This indicates the splitting "scheme": `random`, `modulo` or `contiguous`. |
    | `fold` | The fold number, 0 to 4, that indicates to which cross-validation fold the mutant belongs. |
    """)
    return


@app.cell
def folds_long_cell(df, pl):
    folds_long = (
        df.select("mutant", "random", "modulo", "contiguous")
        .with_columns(
            position=pl.col("mutant").str.extract(r"(\d+)").cast(pl.Int64)
        )
        .unpivot(
            index=["mutant", "position"],
            variable_name="scheme",
            value_name="fold",
        )
    )
    folds_long
    return


@app.cell(hide_code=True)
def ex1_task(mo):
    mo.md(r"""
    Visualize `folds_long` so you can see how each scheme splits up the protein mutants. Use `mark_tick`. Put `position` on the x axis, `scheme` on the y axis, and color each tick by `fold`. Treat `fold` as a category, not a number, with `"fold:N"` when you define the color encoding.

    Which scheme looks like stripes? Which looks like blocks? Which looks like noise?

    <details>
    <summary>Hint</summary>

    ```python
    alt.Chart(folds_long, width=800, height=120).mark_tick(thickness=2).encode(
        x=alt.X("position").title("Position in protein"),
        y=alt.Y("scheme").title(None),
        color=alt.Color("fold:N").title("Fold"),
    )
    ```

    </details>
    """)
    return


@app.cell
def ex1():
    # Code goes here
    return


@app.cell(hide_code=True)
def embed_md(mo):
    mo.md(r"""
    ## Embed every mutant sequence

    This is the same as the last assignment. The cells below load ESMC, embed every sequence in `df`, and set up the features `X` and the target `y`. Just run them.
    """)
    return


@app.cell
def load_model(AutoModel, AutoTokenizer, torch):
    tokenizer = AutoTokenizer.from_pretrained("biohub/ESMC-600M")
    model = (
        AutoModel.from_pretrained("biohub/ESMC-600M", dtype=torch.bfloat16)
        .to("cuda")
        .eval()
    )
    return model, tokenizer


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
def xy_setup(df, embed_sequences):
    X = embed_sequences(df["mutated_sequence"].to_list())
    y = df["DMS_score"].to_numpy()
    X.shape, y.shape
    return


@app.cell(hide_code=True)
def ex2_md(mo):
    mo.md(r"""
    ## Exercise 2: train on four folds, test on one

    Our data is split into five folds for each of the three splitting schemes: "random", "modulo", and "contiguous."

    Write a function `evaluate(model, scheme, test_fold=4)` that:

    1. Finds the rows where `df[scheme]` equals `test_fold`. Those are the test rows. All other rows will be used as training rows.
    2. Trains `model` on the training rows of `X` and `y` with `.fit`.
    3. Predicts the test rows with `.predict`.
    4. Returns the Spearman correlation between the actual and predicted test scores, as a single number.

    **How to pick rows.** Make a boolean array with one `True` per test row: `is_test = (df[scheme] == test_fold).to_numpy()`. Then `X[is_test]` gives the test rows and `X[~is_test]` gives the training rows. The `~` means "not".

    For Spearman, do what you did last time: put actual and predicted in a polars DataFrame and use `pl.corr(..., method="spearman")`.

    To test your function you should be able to run this code:

    ```python
    _mod = RandomForestRegressor()

    evaluate(_mod, scheme="random", test_fold=1)
    ```

    <details>
    <summary>Hint</summary>

    ```python
    def evaluate(model, scheme, test_fold=4):
        # Train on every fold except test_fold, test on test_fold, return Spearman.
        is_test = (df[scheme] == test_fold).to_numpy()
        model.fit(X[~is_test], y[~is_test])
        predicted = model.predict(X[is_test])
        results = pl.DataFrame({"actual": y[is_test], "predicted": predicted})
        return results.select(pl.corr("actual", "predicted", method="spearman")).item()
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
    ## Exercise 3: ridge versus random forest on all three splits

    Two models from scikit-learn. Both are already imported.

    | Tool | What it does | Docs |
    | --- | --- | --- |
    | `RidgeCV(alphas=...)` | Linear regression with a penalty that shrinks the weights. It tries each `alpha` you give it and keeps the best one. Fast. | [RidgeCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html) |
    | `RandomForestRegressor(n_estimators=..., n_jobs=..., random_state=...)` | Many decision trees that vote. Same as last time. | [RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) |
    | `np.logspace(-2, 3, 12)` | 12 numbers spread evenly on a log scale from 0.01 to 1000. A good set of `alpha` values to try. | [logspace](https://numpy.org/doc/stable/reference/generated/numpy.logspace.html) |

    Now do this:

    1. Make a dict `MODELS` with two entries: `"ridge"` and `"random forest"`. Use `RidgeCV(alphas=np.logspace(-2, 3, 12))` and `RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=0)` as the values for "ridge" and "random forest," respectively.
    2. Call `evaluate` for every model and every scheme (`"random"`, `"modulo"` and `"contiguous"`). Keep `test_fold=4`.
    3. Put the results in a polars DataFrame called `results` with columns `scheme`, `model` and `spearman`. Show it.

    The random forest takes 10 to 20 seconds per fit. The whole grid should finish in about a minute.

    <details>
    <summary>Hint</summary>

    ```python
    MODELS = {
        "ridge": RidgeCV(alphas=np.logspace(-2, 3, 12)),
        "random forest": RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=0),
    }
    results = pl.DataFrame(
        [
            {"scheme": scheme, "model": name, "spearman": evaluate(model, scheme)}
            for scheme in ["random", "modulo", "contiguous"]
            for name, model in MODELS.items()
        ]
    )
    results
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
    ## Exercise 4: plot the results

    Make a grouped bar chart of `results`. Put `scheme` on the x axis. Altair sorts categories alphabetically, so set the order yourself with `.scale(domain=["random", "modulo", "contiguous"])`. Use `xOffset="model"` to put the two models side by side inside each group. Put `spearman` on the y axis and color the bars by `model`.

    <details>
    <summary>Hint</summary>

    ```python
    alt.Chart(results, width=300).mark_bar().encode(
        x=alt.X("scheme").title("Split scheme").scale(domain=["random", "modulo", "contiguous"]),
        xOffset="model",
        y=alt.Y("spearman").title("Spearman on the test fold"),
        color="model",
    )
    ```

    </details>
    """)
    return


@app.cell
def ex4():
    # Code goes here
    return


@app.cell(hide_code=True)
def ex5_md(mo):
    mo.md(r"""
    ## Exercise 5: use all five folds

    We have five folds for each splitting scheme. We can use all five folds as test data to get five performance measurements per scheme. In this exercise:

    1. Call `evaluate` for every model, every scheme **and every `test_fold` from 0 to 4**. Put the results in a DataFrame called `cv_results` with columns `scheme`, `model`, `test_fold` and `spearman`. This should have 30 evaluation scores. The random forest models take a while. Expect 3 to 5 minutes.
    2. Group by `scheme` and `model` and compute the mean and standard deviation of `spearman`. Call it `cv_summary` and show it.
    3. Plot every value in `cv_results` as a point. Put `model` on the x axis and `spearman` on the y axis, color by `model`, and make one panel per `scheme` with `column=alt.Column("scheme")`. Points at the same x will sit on top of each other, so add a little random sideways shift (**jitter**). Compute it with `.transform_calculate(jitter=(alt.expr.random() - 0.5) * 0.5)`, then use it as `xOffset=alt.XOffset("jitter:Q").scale(domain=[-0.5, 0.5])`. The `0.5` is the jitter width. The fixed scale domain stops Altair from stretching the jitter to fill the slot.

    <details>
    <summary>Hint</summary>

    ```python
    cv_results = pl.DataFrame(
        [
            {
                "scheme": scheme,
                "model": name,
                "test_fold": fold,
                "spearman": evaluate(model, scheme, test_fold=fold),
            }
            for scheme in ["random", "modulo", "contiguous"]
            for name, model in MODELS.items()
            for fold in range(5)
        ]
    )

    cv_summary = cv_results.group_by("scheme", "model").agg(
        mean=pl.col("spearman").mean(),
        sd=pl.col("spearman").std(),
    )

    _chart = (
        alt.Chart(cv_results, width=120, height=300)
        .mark_circle(size=80, opacity=0.8)
        .transform_calculate(jitter=(alt.expr.random() - 0.5) * 0.5)
        .encode(
            x=alt.X("model").title(None),
            xOffset=alt.XOffset("jitter:Q").scale(domain=[-0.5, 0.5]),
            y=alt.Y("spearman").title("Spearman (one point per test fold)"),
            color=alt.Color("model").legend(None),
            column=alt.Column("scheme").title("Split scheme").sort(["random", "modulo", "contiguous"]),
            tooltip=["scheme", "model", "test_fold", "spearman"],
        )
    )

    mo.vstack([cv_summary, _chart])
    ```

    </details>
    """)
    return


@app.cell
def ex5():
    # Code goes here
    return


@app.cell(hide_code=True)
def ex6_md(mo):
    mo.md(r"""
    ## Exercise 6: what did you learn?

    In the markdown cell below, answer in a few sentences each:

    1. Which split scheme is hardest for the models? Why does that make sense, given how the folds are made?
    2. Which model holds up better when the split gets harder? Look at the gap between the random bar and the contiguous bar for each model. A linear model would have more "inductive bias" than a random forest. Does that help to make sense of the performance trend you observed?
    3. In the last assignment you used a random 80/20 split. Was that Spearman value a fair estimate of how well the model would predict a mutation at a position that is not in the training data?
    """)
    return


@app.cell
def ex6(mo):
    mo.md(r"""
    *put your answer here*
    """)
    return


if __name__ == "__main__":
    app.run()

