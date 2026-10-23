# %% [markdown]
# Source notebook: Lesson_4.ipynb
# Cell order and explanatory text are preserved; saved outputs are omitted.
# See SOURCE_FILES.md at the repository root for runtime requirements.

# %% [markdown]
# # L4: Measuring Search Relevance

# %% [markdown]
# <p style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px"> ⏳ <b>Note <code>(Kernel Starting)</code>:</b> This notebook takes about 30 seconds to be ready to use. You may start and watch the video while you wait.</p>

# %%
# Notebook cell 3
import warnings
warnings.filterwarnings('ignore')

# %%
# Notebook cell 4
import pandas as pd

products_df = pd.read_csv(
    "shared_data/WANDS/product.csv", 
    sep="\t", 
    index_col="product_id", 
    keep_default_na=False,  # some products do not have a description
)
products_df.head()

# %% [markdown]
# <p style="background-color:#fff6ff; padding:15px; border-width:3px; border-color:#efe6ef; border-style:solid; border-radius:6px"> 💻 &nbsp; <b>Access <code>requirements.txt</code> and <code>helper.py</code> files:</b> 1) click on the <em>"File"</em> option on the top menu of the notebook and then 2) click on <em>"Open"</em>. For more help, please see the <em>"Appendix - Tips and Help"</em> Lesson.</p>

# %%
# Notebook cell 6
num_products = 5000

# %%
# Notebook cell 7
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

product_name_embeddings = model.encode(
    products_df["product_name"][0:num_products].tolist()
)
product_name_embeddings.shape

# %%
# Notebook cell 8
product_description_embeddings = model.encode(
    products_df["product_description"][0:num_products].tolist()
)
product_description_embeddings.shape

# %% [markdown]
# ## Building the collection

# %%
# Notebook cell 10
from qdrant_client import QdrantClient, models

client = QdrantClient("http://localhost:6333")
client.delete_collection("wands-products")
client.create_collection(
    collection_name="wands-products",
    vectors_config={
        "product_name": models.VectorParams(
            size=384,
            distance=models.Distance.COSINE,
        ),
        "product_description": models.VectorParams(
            size=384,
            distance=models.Distance.COSINE,
        ),
    },
    optimizers_config=models.OptimizersConfigDiff(
        default_segment_number=2,
        indexing_threshold=1000,
    ),
)

# %%
# Notebook cell 11
client.upload_collection(
    collection_name="wands-products",
    vectors={
        "product_name": product_name_embeddings,
        "product_description": product_description_embeddings,
    },
    payload=products_df.to_dict(orient="records"),
    ids=products_df.index.tolist(),
    batch_size=64,
)

# %%
# Notebook cell 12
client.count("wands-products")

# %%
# Notebook cell 13
import time

time.sleep(1.0)
collection = client.get_collection("wands-products")
while collection.status != models.CollectionStatus.GREEN:
    time.sleep(1.0)
    collection = client.get_collection("wands-products")

collection

# %% [markdown]
# ## Test queries

# %%
# Notebook cell 15
queries_df = pd.read_csv(
    "shared_data/WANDS/query.csv", 
    sep="\t", 
    index_col="query_id",
)
queries_df.head()

# %% [markdown]
# ## Ground truth

# %%
# Notebook cell 17
labels_df = pd.read_csv(
    "shared_data/WANDS/label.csv", 
    sep="\t", 
)
labels_df.sample(n=5)

# %%
# Notebook cell 18
relevancy_scores = {
    "Exact": 10,
    "Partial": 5,
    "Irrelevant": 0,
}

labels_df["score"] = labels_df["label"].map(relevancy_scores.get)
labels_df["query_id"] = labels_df["query_id"].map(lambda x: f"query_{x}")
labels_df["product_id"] = labels_df["product_id"].map(lambda x: f"doc_{x}")
labels_df.sample(n=5)

# %% [markdown]
# ## ranx

# %%
# Notebook cell 20
from ranx import Qrels

qrels = Qrels.from_df(
    labels_df.astype({"query_id": "str", "product_id": "str"}),
    q_id_col="query_id",
    doc_id_col="product_id", 
    score_col="score",
)

# %% [markdown]
# ### Running all the queries

# %%
# Notebook cell 22
queries_df["query_embedding"] = model.encode(
    queries_df["query"].tolist()
).tolist()
queries_df.sample(n=5)

# %%
# Notebook cell 23
from collections import defaultdict

name_run_dict = defaultdict(dict)
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
    )

    for point in results:
        document_id = f"doc_{point.id}"
        name_run_dict[query_id][document_id] = point.score  

name_run_dict

# %%
# Notebook cell 24
from ranx import Run

product_name_run = Run(name_run_dict, name="product_name")

# %%
# Notebook cell 25
description_run_dict = defaultdict(dict)
for id, row in queries_df.iterrows():
    query_id = f"query_{id}"

    results = client.search(
        collection_name="wands-products",
        query_vector=models.NamedVector(
            name="product_description", 
            vector=row["query_embedding"]
        ),
        with_vectors=False,
        with_payload=False,
        limit=100,
    )

    for point in results:
        document_id = f"doc_{point.id}"
        description_run_dict[query_id][document_id] = point.score 

product_description_run = Run(
    description_run_dict, 
    name="product_description"
)

# %%
# Notebook cell 26
from ranx import compare

compare(
    qrels=qrels,
    runs=[
        product_name_run, 
        product_description_run
    ],
    metrics=[
        "precision@10", 
        "recall@10", 
        "mrr@10",
        "dcg@10", 
        "ndcg@10",
    ],
)

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


# %%
# Notebook cell 33

