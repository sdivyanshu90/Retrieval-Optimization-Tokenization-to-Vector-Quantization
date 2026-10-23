# %% [markdown]
# Source notebook: Lesson_5.ipynb
# Cell order and explanatory text are preserved; saved outputs are omitted.
# See SOURCE_FILES.md at the repository root for runtime requirements.

# %% [markdown]
# # L5: Optimizing HNSW Search

# %% [markdown]
# <p style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px"> ⏳ <b>Note <code>(Kernel Starting)</code>:</b> This notebook takes about 30 seconds to be ready to use. You may start and watch the video while you wait.</p>

# %%
# Notebook cell 3
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# <p style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px"> ⏳ <b>Note <code>(Loading the collection)</code>:</b> The following code block might take a few minutes to complete.</p>

# %%
# Notebook cell 5
from qdrant_client import QdrantClient, models

client = QdrantClient("http://localhost:6333", timeout=600)
client.delete_collection("wands-products")
client.recover_snapshot(
    "wands-products", 
    "https://storage.googleapis.com/deeplearning-course-c1/snapshots/wands-products.snapshot",
)
collection = client.get_collection("wands-products")
collection

# %% [markdown]
# <p style="background-color:#fff6ff; padding:15px; border-width:3px; border-color:#efe6ef; border-style:solid; border-radius:6px"> 💻 &nbsp; <b>Access <code>requirements.txt</code> and <code>helper.py</code> files:</b> 1) click on the <em>"File"</em> option on the top menu of the notebook and then 2) click on <em>"Open"</em>. For more help, please see the <em>"Appendix - Tips and Help"</em> Lesson.</p>

# %% [markdown]
# ## HNSW parameters

# %%
# Notebook cell 8
collection.config.hnsw_config

# %% [markdown]
# ## Test queries

# %%
# Notebook cell 10
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

# %%
# Notebook cell 11
import pandas as pd

queries_df = pd.read_csv(
    "shared_data/WANDS/query.csv", 
    sep="\t", 
    index_col="query_id",
)
queries_df["query_embedding"] = model.encode(
    queries_df["query"].tolist()
).tolist()
queries_df.sample(n=5)

# %% [markdown]
# ## ANN search

# %%
# Notebook cell 13
client.search(
    "wands-products",
    query_vector=models.NamedVector(
        name="product_name",
        vector=model.encode(queries_df.loc[0, "query"])
    ),
    limit=3,
    with_vectors=False,
    with_payload=False,
)

# %% [markdown]
# ## kNN search

# %%
# Notebook cell 15
client.search(
    "wands-products",
    query_vector=models.NamedVector(
        name="product_name",
        vector=model.encode(queries_df.loc[0, "query"])
    ),
    limit=3,
    with_vectors=False,
    with_payload=False,
    search_params=models.SearchParams(
        exact=True,  # Turns on the exact search mode
    ),
)

# %% [markdown]
# ### Ground truth

# %%
# Notebook cell 17
from collections import defaultdict
from ranx import Qrels

knn_qrels_dict = defaultdict(dict)
for id, row in queries_df.iterrows():
    query_id = f"query_{id}"

    results = client.search(
        collection_name="wands-products",
        query_vector=models.NamedVector(
            name="product_name", 
            vector=row["query_embedding"]
        ),
        with_vectors=False,
        with_payload=False,
        limit=100,
        search_params=models.SearchParams(
            exact=True,  # enable exact search
        ),
    )

    for point in results:
        document_id = f"doc_{point.id}"
        # The conversion to integer is required because ranx expects integers
        knn_qrels_dict[query_id][document_id] = int(point.score * 100)

qrels = Qrels(knn_qrels_dict)
qrels

# %% [markdown]
# ### ANN search

# %%
# Notebook cell 19
from ranx import Run

run_dict = defaultdict(dict)
for id, row in queries_df.iterrows():
    query_id = f"query_{id}"

    results = client.search(
        collection_name="wands-products",
        query_vector=models.NamedVector(
            name="product_name", 
            vector=row["query_embedding"]
        ),
        with_vectors=False,
        with_payload=False,
        limit=100,
        search_params=models.SearchParams(
            exact=False,  # disable exact search
        ),
    )

    for point in results:
        document_id = f"doc_{point.id}"
        run_dict[query_id][document_id] = point.score

initial_run = Run(
    run_dict, 
    name="initial",
)
initial_run

# %%
# Notebook cell 20
from ranx import evaluate

evaluate(
    qrels=qrels, 
    run=initial_run, 
    metrics=["precision@25"]
)

# %% [markdown]
# ## Tweaking the HNSW parameters

# %%
# Notebook cell 22
client.update_collection(
    collection_name="wands-products",
    hnsw_config=models.HnswConfigDiff(
        m=64, 
        ef_construct=200,
    )
)

# %%
# Notebook cell 23
import time

time.sleep(1.0)
collection = client.get_collection("wands-products")
while collection.status != models.CollectionStatus.GREEN:
    time.sleep(1.0)
    collection = client.get_collection("wands-products")

collection

# %%
# Notebook cell 24
tweaked_run_dict = defaultdict(dict)
for id, row in queries_df.iterrows():
    query_id = f"query_{id}"

    results = client.search(
        collection_name="wands-products",
        query_vector=models.NamedVector(
            name="product_name", 
            vector=row["query_embedding"]
        ),
        with_vectors=False,
        with_payload=False,
        limit=100,
        search_params=models.SearchParams(
            exact=False,  # disable exact search
        ),
    )

    for point in results:
        document_id = f"doc_{point.id}"
        tweaked_run_dict[query_id][document_id] = point.score

tweaked_run = Run(
    tweaked_run_dict, 
    name="tweaked"
)
tweaked_run

# %%
# Notebook cell 25
evaluate(
    qrels=qrels, 
    run=tweaked_run, 
    metrics=["precision@25"]
)

# %%
# Notebook cell 26


# %%
# Notebook cell 27


# %%
# Notebook cell 28


# %%
# Notebook cell 29


# %%
# Notebook cell 30


# %%
# Notebook cell 31


# %%
# Notebook cell 32

