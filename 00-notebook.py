# /// script
# dependencies = [
#     "altair==6.3.0",
#     "marimo>=0.25.0",
#     "polars==1.44.2",
#     "remotezip==0.12.6",
#     "vegafusion==2.0.3",
#     "vl-convert-python==1.9.0.post1",
# ]
# ///

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import polars as pl
    import altair as alt
    from remotezip import RemoteZip

    # Some ProteinGym datasets have far more than altair's default 5000-row limit.
    alt.data_transformers.enable("vegafusion")
    return RemoteZip, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Unit 6 assignment

    In this assignment you will explore one **deep mutational scanning (DMS)** dataset from [ProteinGym](https://proteingym.org). A DMS experiment makes many single-amino-acid mutants of one protein and measures how well each mutant works. ProteinGym collects hundreds of these experiments into one benchmark.

    Each dataset is a table with one row per mutant. The two columns we care about are:

    | Column | Description |
    | --- | --- |
    | `mutant` | The mutation ID, like `A192M`: wild-type amino acid `A`, position `192`, mutant amino acid `M`. Multi-mutants are joined with `:` (for example `A192M:L10P`). |
    | `DMS_score` | The experimental score for that mutant. What it measures depends on the assay. You will find out what it means for your dataset at the end. |

    ### Pick a dataset

    1. Go to [ProteinGym's benchmark table](https://proteingym.org/benchmarks).
    2. Make sure the first dropdown says **DMS Substitution**.
    3. Click **Individual View** (top right).
    4. Copy any value from the **DMS ID** column and paste it into `DATASET_ID` below.
    """)
    return


@app.cell
def _():
    DATASET_ID = "Q837P4_ENTFA_Meier_2023"
    return (DATASET_ID,)


@app.cell
def _(DATASET_ID, RemoteZip, pl):
    PROTEINGYM_SUBSTITUTIONS_URL = (
        "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/"
        "DMS_ProteinGym_substitutions.zip"
    )

    with RemoteZip(PROTEINGYM_SUBSTITUTIONS_URL) as _zf:
        _match = next(n for n in _zf.namelist() if DATASET_ID in n)
        with _zf.open(_match) as _f:
            df = pl.read_csv(_f)

    df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Histogram of DMS_score
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Plot a histogram of `DMS_score` from `df`. Bin it into about 40 bins.

    <details>
    <summary>Hint</summary>

    ```python
    alt.Chart(df, width=600).mark_bar().encode(
        x=alt.X('DMS_score').bin(maxbins=40),
        y=alt.Y('count()'),
    )
    ```

    </details>
    """)
    return


@app.cell
def _():
    # Code goes here
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Mutants per position along the protein
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    How many mutants does the dataset have at each position in the protein? To answer this you need a `position` column.

    1. Drop the multi-mutants (rows where `mutant` contains `:`).
    2. Add three columns from the `mutant` string: `wildtype` (first letter), `position` (the number), and `mutant_aa` (last letter). Call the result `df_parsed`.
    3. Plot a bar chart of the number of mutants at each `position`.

    Write the code yourself. A working example is in the hint if you get stuck.

    <details>
    <summary>Hint</summary>

    ```python
    df_parsed = (
        df.filter(~pl.col('mutant').str.contains(':'))
        .with_columns(
            wildtype=pl.col('mutant').str.slice(0, 1),
            position=pl.col('mutant').str.extract(r'(\d+)').cast(pl.Int64),
            mutant_aa=pl.col('mutant').str.slice(-1),
        )
    )

    alt.Chart(df_parsed, width=1000).mark_bar().encode(
        x=alt.X('position'),
        y=alt.Y('count()'),
    )
    ```

    </details>
    """)
    return


@app.cell
def _():
    # Code goes here
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Top 10 scoring mutations
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Find the 10 mutants with the highest `DMS_score` in `df_parsed`. Polars has a [`top_k`](https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.top_k.html) method that does this in one step.

    Show them on a **scatter chart**: one point per mutant, placed at its `position` along the protein, with the point color set by the mutant amino acid (`mutant_aa`).

    <details>
    <summary>Hint</summary>

    ```python
    top10 = df_parsed.top_k(10, by='DMS_score')

    alt.Chart(top10, width=1000).mark_point(size=100, filled=True).encode(
        x=alt.X('position').scale(domain=[1, df_parsed['position'].max()]),
        y=alt.Y('DMS_score'),
        color=alt.Color('mutant_aa').title('Mutant amino acid'),
        tooltip=['mutant', 'DMS_score'],
    )
    ```

    </details>
    """)
    return


@app.cell
def _():
    # Code goes here
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What does DMS_score mean?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Every ProteinGym dataset comes from a published experiment. The `DMS_score` means something different in each one (growth rate, binding, fluorescence, stability, and so on).

    Use an LLM (molab's built-in agent, ChatGPT, Claude, Gemini, ...) to find the source paper for your dataset and learn what the score measures. Here is an example prompt. Replace the dataset ID with your own.

    > I am working with the ProteinGym DMS substitution dataset `Q837P4_ENTFA_Meier_2023`. Find the original paper that produced this deep mutational scanning data. Give me the title, first author, year, and a link (DOI or journal). Then explain, in plain terms: what experiment was done, what the `DMS_score` column measures, and whether a higher score means the mutant is better or worse. Tell me which parts you are unsure about. Use ASD-STE100 (Simplified Technical English) rules for wording: one idea per sentence, active voice, approved simple words only, no jargon, no strung-together nouns.

    LLMs can make up citations. Open the link and check that the paper is real and that it matches your dataset.

    In the markdown cell below, write:

    1. The paper citation (title, first author, year, link).
    2. What `DMS_score` measures in this dataset.
    3. Whether a higher score is better or worse for the protein.
    4. One sentence: did the LLM's answer check out against the paper?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    *put your answer here*
    """)
    return


if __name__ == "__main__":
    app.run()
