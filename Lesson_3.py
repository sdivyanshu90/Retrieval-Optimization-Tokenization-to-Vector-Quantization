# %% [markdown]
# Source notebook: Lesson_3.ipynb
# Cell order and explanatory text are preserved; saved outputs are omitted.
# See SOURCE_FILES.md at the repository root for runtime requirements.

# %% [markdown]
# # L3: Practical Implications of the Tokenization

# %% [markdown]
# <p style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px"> ⏳ <b>Note <code>(Kernel Starting)</code>:</b> This notebook takes about 30 seconds to be ready to use. You may start and watch the video while you wait.</p>

# %%
# Notebook cell 3
import warnings
warnings.filterwarnings('ignore')

# %%
# Notebook cell 4
from sentence_transformers import SentenceTransformer, util

sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
out_vector = sbert_model.encode("Vector search optimization")
out_vector.shape

# %% [markdown]
# <p style="background-color:#fff6ff; padding:15px; border-width:3px; border-color:#efe6ef; border-style:solid; border-radius:6px"> 💻 &nbsp; <b>Access <code>requirements.txt</code> and <code>helper.py</code> files:</b> 1) click on the <em>"File"</em> option on the top menu of the notebook and then 2) click on <em>"Open"</em>. For more help, please see the <em>"Appendix - Tips and Help"</em> Lesson.</p>

# %%
# Notebook cell 6
sbert_tokenizer = sbert_model.tokenizer._tokenizer
sbert_tokenizer

# %% [markdown]
# ### Unknown tokens

# %%
# Notebook cell 8
sbert_tokenizer.encode("I feel 😊").tokens

# %%
# Notebook cell 9
sbert_tokenizer.encode("I feel happy").tokens

# %%
# Notebook cell 10
sbert_tokenizer.encode("I feel 🙁").tokens

# %%
# Notebook cell 11
sbert_tokenizer.encode("I feel sad").tokens

# %% [markdown]
# ### Identifiers

# %%
# Notebook cell 13
sbert_tokenizer.encode("Broadcom BCM2712").tokens

# %% [markdown]
# ### Typos

# %%
# Notebook cell 15
sentences = [
    "Great accommodation",
    "Great acommodation",
]
for sentence in sentences:
    print(sbert_tokenizer.encode(sentence).tokens)

# %% [markdown]
# ### Numerical values and date/time

# %%
# Notebook cell 17
sentences = [
    "This shirt costs $55.",
    "This shirt costs fifty five dollars.",
    "This shirt costs $50.",
    "This shirt costs $559.",
    "This shirt has a 10% discount from $60.",
]
for sentence in sentences:
    print(sbert_tokenizer.encode(sentence).tokens)

# %%
# Notebook cell 18
sentences = [
    "16th February 2024",
    "2024-02-16",
    "17th February 2024",
    "18th February 2024",
    "19th February 2024",
    "20th February 2024",
    "15th February 2024",
]
for sentence in sentences:
    print(sbert_tokenizer.encode(sentence).tokens)

# %% [markdown]
# ## Implications on semantic similarity

# %%
# Notebook cell 20
import plotly.express as px

sentences = [
    "I feel 😊",
    "I feel happy",
    "I feel 🙁",
    "I feel sad",
]
embeddings = sbert_model.encode(sentences)
cosine_scores = util.cos_sim(embeddings, embeddings)

px.imshow(
    cosine_scores,
    x=sentences,
    y=sentences,
    text_auto=True,
)

# %% [markdown]
# ### Typos

# %%
# Notebook cell 22
sentences = [
    "Great accommodation",
    "Great acommodation",
]
embeddings = sbert_model.encode(sentences)
cosine_scores = util.cos_sim(embeddings, embeddings)
px.imshow(
    cosine_scores,
    x=sentences,
    y=sentences,
    text_auto=True,
)

# %% [markdown]
# ### Numerical values and date/time

# %%
# Notebook cell 24
sentences = [
    "This shirt costs $55.",
    "This shirt costs fifty five dollars.",
    "This shirt costs $50.",
    "This shirt costs $559.",
    "This shirt has a 10% discount from $60.",
]
embeddings = sbert_model.encode(sentences)
cosine_scores = util.cos_sim(embeddings, embeddings)
fig = px.imshow(
    cosine_scores,
    x=sentences,
    y=sentences,
    text_auto=True,
)
fig.update_xaxes(tickangle=-30)

# %%
# Notebook cell 25
sentences = [
    "16th February 2024",
    "2024-02-16",
    "17th February 2024",
    "18th February 2024",
    "19th February 2024",
    "20th February 2024",
    "15th February 2024",
]
embeddings = sbert_model.encode(sentences)
cosine_scores = util.cos_sim(embeddings, embeddings)
fig = px.imshow(
    cosine_scores,
    x=sentences,
    y=sentences,
    text_auto=True,
)
fig.update_layout(
    xaxis={"type": "category"}, 
    yaxis={"type": "category"}
)
fig.update_xaxes(tickangle=-30)

# %% [markdown]
# ## Impact on different models

# %% [markdown]
# ### OpenAI

# %%
# Notebook cell 28
import tiktoken

openai_tokenizer = tiktoken.encoding_for_model("text-embedding-3-large")
openai_tokenizer.n_vocab

# %%
# Notebook cell 29
token_ids = openai_tokenizer.encode("I feel 😊")
openai_tokenizer.decode_tokens_bytes(token_ids)

# %%
# Notebook cell 30
token_ids = openai_tokenizer.encode("I feel happy")
openai_tokenizer.decode_tokens_bytes(token_ids)

# %%
# Notebook cell 31
token_ids = openai_tokenizer.encode("Broadcom BCM2712")
openai_tokenizer.decode_tokens_bytes(token_ids)

# %%
# Notebook cell 32
sentences = [
    "Great accommodation",
    "Great acommodation",
]
for sentence in sentences:
    token_ids = openai_tokenizer.encode(sentence)
    print(openai_tokenizer.decode_tokens_bytes(token_ids))

# %%
# Notebook cell 33
sentences = [
    "This shirt costs $55.",
    "This shirt costs fifty five dollars.",
    "This shirt costs $50.",
    "This shirt costs $559.",
    "This shirt has a 10% discount from $60.",
]
for sentence in sentences:
    token_ids = openai_tokenizer.encode(sentence)
    print(openai_tokenizer.decode_tokens_bytes(token_ids))

# %%
# Notebook cell 34
sentences = [
    "16th February 2024",
    "2024-02-16",
    "17th February 2024",
    "18th February 2024",
    "19th February 2024",
    "20th February 2024",
    "15th February 2024",
]
for sentence in sentences:
    token_ids = openai_tokenizer.encode(sentence)
    print(openai_tokenizer.decode_tokens_bytes(token_ids))

# %% [markdown]
# #### Vector similarity

# %%
# Notebook cell 36
from openai import OpenAI
from helper import get_openai_api_key

openai_client = OpenAI(api_key=get_openai_api_key())

# %%
# Notebook cell 37
sentences = [
    "I feel 😊",
    "I feel happy",
    "I feel 🙁",
    "I feel sad",
]
embeddings = [
    embedding.embedding
    for embedding in openai_client.embeddings.create(
        input=sentences, model="text-embedding-3-large"
    ).data
]
cosine_scores = util.cos_sim(embeddings, embeddings)
px.imshow(
    cosine_scores,
    x=sentences,
    y=sentences,
    text_auto=True,
)

# %%
# Notebook cell 38
sentences = [
    "Great accommodation",
    "Great acommodation",
]
embeddings = [
    embedding.embedding
    for embedding in openai_client.embeddings.create(
        input=sentences, model="text-embedding-3-large"
    ).data
]
cosine_scores = util.cos_sim(embeddings, embeddings)
fig = px.imshow(
    cosine_scores,
    x=sentences,
    y=sentences,
    text_auto=True,
)
fig.update_xaxes(tickangle=-30)

# %%
# Notebook cell 39
sentences = [
    "This shirt costs $55.",
    "This shirt costs fifty five dollars.",
    "This shirt costs $50.",
    "This shirt costs $559.",
    "This shirt has a 10% discount from $60.",
]
embeddings = [
    embedding.embedding
    for embedding in openai_client.embeddings.create(
        input=sentences, model="text-embedding-3-large"
    ).data
]
cosine_scores = util.cos_sim(embeddings, embeddings)
fig = px.imshow(
    cosine_scores,
    x=sentences,
    y=sentences,
    text_auto=True,
)
fig.update_xaxes(tickangle=-30)

# %%
# Notebook cell 40
sentences = [
    "16th February 2024",
    "2024-02-16",
    "17th February 2024",
    "18th February 2024",
    "19th February 2024",
    "20th February 2024",
    "15th February 2024",
]
embeddings = [
    embedding.embedding
    for embedding in openai_client.embeddings.create(
        input=sentences, model="text-embedding-3-large"
    ).data
]
cosine_scores = util.cos_sim(embeddings, embeddings)

fig = px.imshow(
    cosine_scores,
    x=sentences,
    y=sentences,
    text_auto=True,
)
fig.update_layout(
    xaxis={"type": "category"}, 
    yaxis={"type": "category"}
)
fig.update_xaxes(tickangle=-30)

# %% [markdown]
# ## Vector search in practice

# %%
# Notebook cell 42
from qdrant_client import QdrantClient, models

client = QdrantClient("http://localhost:6333")
client.get_collections()

# %%
# Notebook cell 43
examples = [
    ("Sleeveless maxi dress with a V-neckline and wrap front."
     "Comes with a tie belt at the waist.", 29.95),
    ("Slim-fit jeans in washed stretch denim with a button fly "
     "and tapered legs.", 39.99),
    ("Double-breasted blazer in textured-weave fabric with peak "
     "lapels and front flap pockets.", 59.50),
    ("Lightweight bomber jacket with ribbed cuffs, a baseball collar, "
     "and zip front closure.", 45.00),
    ("Chunky knit sweater with dropped shoulders and ribbed trim around "
     "the neck, cuffs, and hem.", 25.99),
    ("Tailored trousers in a smooth woven fabric with a concealed "
     "hook-and-eye closure and welt back pockets.", 34.99),
    ("Classic trench coat with adjustable belt, storm flap, and button "
     "front closure.", 79.90),
    ("High-rise pencil skirt in a stretch fabric with a hidden rear zip "
     "and back slit.", 22.99),
    ("Athletic-fit polo shirt in moisture-wicking fabric with a ribbed "
     "collar and two-button placket.", 19.95),
    ("Soft flannel pajama set with long sleeves, matching pants, and a "
     "comfortable elastic waistband.", 32.00),
    ("Quilted puffer jacket with a detachable hood and zippered side "
     "pockets.", 48.99),
    ("Cropped denim jacket with distressed details and button-flap chest "
     "pockets.", 36.50),
    ("Fitted bodysuit with a scoop neckline and snap-button closure at "
     "the bottom.", 15.99),
    ("Lightly padded parka with a faux fur-lined hood, drawstring waist, "
     "and snap front pockets.", 69.95),
    ("Mesh panel sports leggings with a high waist and reflective details "
     "for nighttime visibility.", 27.99),
    ("Button-up cardigan in a soft knit with long sleeves and ribbed "
     "trim.", 24.50),
    ("Leather moto jacket with zippered cuffs, a notched collar, and "
     "asymmetrical zip closure.", 95.00),
    ("Velvet slip dress with a lace trim neckline and adjustable "
     "spaghetti straps.", 31.99),
    ("Cargo shorts with multiple pockets and a durable belt loop "
     "waistband.", 22.95),
    ("Wide-leg palazzo pants with a high-rise fit and side zip "
     "closure.", 38.99),
    ("Graphic print tee featuring an original artwork design and classic "
     "crew neck.", 14.99),
    ("Boho-style maxi skirt with an elastic waistband and tiered ruffle "
     "detailing.", 33.50),
    ("Men's linen shirt with a Mandarin collar and buttoned chest "
     "pocket.", 29.95),
    ("Cable knit beanie with a fold-over cuff and soft fleece "
     "lining.", 12.99),
    ("Sequin cocktail dress with a plunging V-neck and bodycon "
     "fit.", 49.99),
]

# %%
# Notebook cell 44
client.delete_collection("clothes")
client.create_collection(
    "clothes",
    vectors_config=models.VectorParams(
        size=384,
        distance=models.Distance.COSINE,
    )
)

# %%
# Notebook cell 45
import uuid

client.upsert(
    "clothes",
    points=[
        models.PointStruct(
            id=uuid.uuid4().hex,
            vector=sbert_model.encode(description),
            payload={"description": description, "price": price},
        )
        for description, price in examples
    ]
)

# %%
# Notebook cell 46
client.search(
    "clothes",
    query_vector=sbert_model.encode("for cold weather"),
    with_payload=True,
    limit=3,
)

# %% [markdown]
# ### Vector search with additional constraints

# %%
# Notebook cell 48
client.search(
    "clothes",
    query_vector=sbert_model.encode("for cold weather under $40"),
    with_payload=True,
    limit=3,
)

# %%
# Notebook cell 49
client.create_payload_index(
    "clothes",
    field_name="price",
    field_schema=models.PayloadSchemaType.FLOAT,
)

# %%
# Notebook cell 50
client.search(
    "clothes",
    query_vector=sbert_model.encode("for cold weather"),
    with_payload=True,
    limit=3,
    query_filter=models.Filter(
        must=[
            models.FieldCondition(
                key="price",
                range=models.Range(
                    lte=40.0,
                )
            )
        ]
    )
)

# %%
# Notebook cell 51


# %%
# Notebook cell 52


# %%
# Notebook cell 53


# %%
# Notebook cell 54


# %%
# Notebook cell 55


# %%
# Notebook cell 56


# %%
# Notebook cell 57


# %%
# Notebook cell 58

