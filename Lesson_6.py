# %% [markdown]
# Source notebook: Lesson_6.ipynb
# Cell order and explanatory text are preserved; saved outputs are omitted.
# See SOURCE_FILES.md at the repository root for runtime requirements.

# %% [markdown]
# # L6: Vector Quantization

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
# ## Test queries

# %%
# Notebook cell 8
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

# %%
# Notebook cell 9
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

# %%
# Notebook cell 10
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
        limit=50,
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
# ## Product Quantization (PQ)

# %%
# Notebook cell 12
client.update_collection(
    "wands-products",
    quantization_config=models.ProductQuantization(
        product=models.ProductQuantizationConfig(
            compression=models.CompressionRatio.X64,
            always_ram=True,
        )    
    ),
)

# %%
# Notebook cell 13
import time

time.sleep(1.0)
collection = client.get_collection("wands-products")
while collection.status != models.CollectionStatus.GREEN:
    time.sleep(1.0)
    collection = client.get_collection("wands-products")

collection

# %%
# Notebook cell 14
import numpy as np

response_times = []
pq_run_dict = defaultdict(dict)
for id, row in queries_df.iterrows():
    query_id = f"query_{id}"

    # Measure the initial time
    start_time = time.monotonic()

    results = client.search(
        collection_name="wands-products",
        query_vector=models.NamedVector(
            name="product_name", 
            vector=row["query_embedding"]
        ),
        search_params=models.SearchParams(
            quantization=models.QuantizationSearchParams(
                rescore=False # Disable re-scoring using the original vectors
            )
        ),
        with_vectors=False,
        with_payload=False,
        limit=50,
    )

    # Store the response time in the list
    response_times.append(time.monotonic() - start_time)

    for point in results:
        document_id = f"doc_{point.id}"
        pq_run_dict[query_id][document_id] = point.score

np.mean(response_times)

# %%
# Notebook cell 15
from ranx import Run, evaluate

product_name_pq_run = Run(
    pq_run_dict, 
    name="product_name_pq"
)
evaluate(
    qrels=qrels, 
    run=product_name_pq_run, 
    metrics=["precision@25"]
)

# %%
# Notebook cell 16
response_times = []
pq_rescore_run_dict = defaultdict(dict)
for id, row in queries_df.iterrows():
    query_id = f"query_{id}"

    start_time = time.monotonic()

    results = client.search(
        collection_name="wands-products",
        query_vector=models.NamedVector(
            name="product_name", 
            vector=row["query_embedding"]
        ),
        search_params=models.SearchParams(
            quantization=models.QuantizationSearchParams(
                rescore=True # Enable re-scoring using the original vectors
            )
        ),
        with_vectors=False,
        with_payload=False,
        limit=50,
    )

    response_times.append(time.monotonic() - start_time)

    for point in results:
        document_id = f"doc_{point.id}"
        pq_rescore_run_dict[query_id][document_id] = point.score

np.mean(response_times)

# %%
# Notebook cell 17
product_name_pq_rescore_run = Run(
    pq_rescore_run_dict,
    name="product_name_pq_rescore"
)
evaluate(
    qrels=qrels, 
    run=product_name_pq_rescore_run, 
    metrics=["precision@25"]
)

# %% [markdown]
# ## Scalar Quantization (SQ)

# %%
# Notebook cell 19
client.update_collection(
    "wands-products",
    quantization_config=models.ScalarQuantization(
        scalar=models.ScalarQuantizationConfig(
            type=models.ScalarType.INT8,
            always_ram=True,
        )    
    ),
)

# %%
# Notebook cell 20
time.sleep(1.0)
collection = client.get_collection("wands-products")
while collection.status != models.CollectionStatus.GREEN:
    time.sleep(1.0)
    collection = client.get_collection("wands-products")

collection

# %%
# Notebook cell 21
response_times = []
sq_run_dict = defaultdict(dict)
for id, row in queries_df.iterrows():
    query_id = f"query_{id}"

    start_time = time.monotonic()

    results = client.search(
        collection_name="wands-products",
        query_vector=models.NamedVector(
            name="product_name", 
            vector=row["query_embedding"]
        ),
        search_params=models.SearchParams(
            quantization=models.QuantizationSearchParams(
                rescore=False # Disable re-scoring using the original vectors
            )
        ),
        with_vectors=False,
        with_payload=False,
        limit=50,
    )

    response_times.append(time.monotonic() - start_time)

    for point in results:
        document_id = f"doc_{point.id}"
        sq_run_dict[query_id][document_id] = point.score

np.mean(response_times)

# %%
# Notebook cell 22
product_name_sq_run = Run(
    sq_run_dict, 
    name="product_name_sq"
)
evaluate(
    qrels=qrels, 
    run=product_name_sq_run, 
    metrics=["precision@25"]
)

# %%
# Notebook cell 23
response_times = []
sq_rescore_run_dict = defaultdict(dict)
for id, row in queries_df.iterrows():
    query_id = f"query_{id}"

    start_time = time.monotonic()

    results = client.search(
        collection_name="wands-products",
        query_vector=models.NamedVector(
            name="product_name", 
            vector=row["query_embedding"]
        ),
        search_params=models.SearchParams(
            quantization=models.QuantizationSearchParams(
                rescore=True # Enable re-scoring using the original vectors
            )
        ),
        with_vectors=False,
        with_payload=False,
        limit=50,
    )

    response_times.append(time.monotonic() - start_time)

    for point in results:
        document_id = f"doc_{point.id}"
        sq_rescore_run_dict[query_id][document_id] = point.score

np.mean(response_times)

# %%
# Notebook cell 24
product_name_sq_rescore_run = Run(
    sq_rescore_run_dict, 
    name="product_name_sq_rescore"
)
evaluate(
    qrels=qrels, 
    run=product_name_sq_rescore_run, 
    metrics=["precision@25"]
)

# %% [markdown]
# ## Binary Quantization (BQ)

# %%
# Notebook cell 26
client.update_collection(
    "wands-products",
    quantization_config=models.BinaryQuantization(
        binary=models.BinaryQuantizationConfig(
            always_ram=True,
        )    
    ),
)

# %%
# Notebook cell 27
time.sleep(1.0)
collection = client.get_collection("wands-products")
while collection.status != models.CollectionStatus.GREEN:
    time.sleep(1.0)
    collection = client.get_collection("wands-products")

collection

# %%
# Notebook cell 28
response_times = []
bq_run_dict = defaultdict(dict)
for id, row in queries_df.iterrows():
    query_id = f"query_{id}"

    start_time = time.monotonic()

    results = client.search(
        collection_name="wands-products",
        query_vector=models.NamedVector(
            name="product_name", 
            vector=row["query_embedding"]
        ),
        search_params=models.SearchParams(
            quantization=models.QuantizationSearchParams(
                rescore=False # Disable re-scoring using the original vectors
            )
        ),
        with_vectors=False,
        with_payload=False,
        limit=50,
    )

    response_times.append(time.monotonic() - start_time)

    for point in results:
        document_id = f"doc_{point.id}"
        bq_run_dict[query_id][document_id] = point.score

np.mean(response_times)

# %%
# Notebook cell 29
product_name_bq_run = Run(
    bq_run_dict, 
    name="product_name_bq"
)
evaluate(
    qrels=qrels, 
    run=product_name_bq_run, 
    metrics=["precision@25"]
)

# %%
# Notebook cell 30
response_times = []
bq_rescore_run_dict = defaultdict(dict)
for id, row in queries_df.iterrows():
    query_id = f"query_{id}"

    start_time = time.monotonic()

    results = client.search(
        collection_name="wands-products",
        query_vector=models.NamedVector(
            name="product_name", 
            vector=row["query_embedding"]
        ),
        search_params=models.SearchParams(
            quantization=models.QuantizationSearchParams(
                rescore=True # Enable re-scoring using the original vectors
            )
        ),
        with_vectors=False,
        with_payload=False,
        limit=50,
    )

    response_times.append(time.monotonic() - start_time)

    for point in results:
        document_id = f"doc_{point.id}"
        bq_rescore_run_dict[query_id][document_id] = point.score

np.mean(response_times)

# %%
# Notebook cell 31
product_name_bq_rescore_run = Run(
    bq_rescore_run_dict, 
    name="product_name_bq_rescore"
)
evaluate(
    qrels=qrels, 
    run=product_name_bq_rescore_run, 
    metrics=["precision@25"]
)

# %%
# Notebook cell 32
from ranx import compare

compare(
    qrels=qrels,
    runs=[
        product_name_pq_run, 
        product_name_pq_rescore_run, 
        product_name_sq_run, 
        product_name_sq_rescore_run, 
        product_name_bq_run, 
        product_name_bq_rescore_run,
    ],
    metrics=["precision@25"],
)

# %%
# Notebook cell 33


# %%
# Notebook cell 34


# %%
# Notebook cell 35


# %%
# Notebook cell 36


# %%
# Notebook cell 37


# %%
# Notebook cell 38


# %%
# Notebook cell 39

