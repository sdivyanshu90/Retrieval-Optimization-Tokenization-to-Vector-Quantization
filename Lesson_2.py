# %% [markdown]
# Source notebook: Lesson_2.ipynb
# Cell order and explanatory text are preserved; saved outputs are omitted.
# See SOURCE_FILES.md at the repository root for runtime requirements.

# %% [markdown]
# # L2: Role of the Tokenizers

# %% [markdown]
# <p style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px"> ⏳ <b>Note <code>(Kernel Starting)</code>:</b> This notebook takes about 30 seconds to be ready to use. You may start and watch the video while you wait.</p>

# %%
# Notebook cell 3
import warnings
warnings.filterwarnings('ignore')

# %%
# Notebook cell 4
training_data = [
    "walker walked a long walk",
]

# %% [markdown]
# ## BPE - Byte-Pair Encoding

# %%
# Notebook cell 6
from tokenizers.trainers import BpeTrainer
from tokenizers.models import BPE
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Whitespace

bpe_tokenizer = Tokenizer(BPE())
bpe_tokenizer.pre_tokenizer = Whitespace()

bpe_trainer = BpeTrainer(vocab_size=14)

# %% [markdown]
# <p style="background-color:#fff6ff; padding:15px; border-width:3px; border-color:#efe6ef; border-style:solid; border-radius:6px"> 💻 &nbsp; <b>Access <code>requirements.txt</code> and <code>helper.py</code> files:</b> 1) click on the <em>"File"</em> option on the top menu of the notebook and then 2) click on <em>"Open"</em>. For more help, please see the <em>"Appendix - Tips and Help"</em> Lesson.</p>

# %%
# Notebook cell 8
bpe_tokenizer.train_from_iterator(training_data, bpe_trainer)

# %%
# Notebook cell 9
bpe_tokenizer.get_vocab()

# %%
# Notebook cell 10
bpe_tokenizer.encode("walker walked a long walk").tokens

# %%
# Notebook cell 11
bpe_tokenizer.encode("wlk").ids

# %%
# Notebook cell 12
bpe_tokenizer.encode("wlk").tokens

# %%
# Notebook cell 13
bpe_tokenizer.encode("she walked").tokens

# %% [markdown]
# ## WordPiece

# %%
# Notebook cell 15
from real_wordpiece.trainer import RealWordPieceTrainer
from tokenizers.models import WordPiece

real_wordpiece_tokenizer = Tokenizer(WordPiece())
real_wordpiece_tokenizer.pre_tokenizer = Whitespace()

real_wordpiece_trainer = RealWordPieceTrainer(
    vocab_size=27,
)

# %%
# Notebook cell 16
real_wordpiece_trainer.train_tokenizer(
    training_data, real_wordpiece_tokenizer
)
real_wordpiece_tokenizer.get_vocab()

# %%
# Notebook cell 17
real_wordpiece_tokenizer.encode("walker walked a long walk").tokens

# %%
# Notebook cell 18
real_wordpiece_tokenizer.encode("wlk").tokens

# %% [markdown]
# **Unknown Characters:**
# The following line will produce an error because it contains unknown characters. Please uncomment the line and run it to see the error.

# %%
# Notebook cell 20
#real_wordpiece_tokenizer.encode("she walked").tokens

# %% [markdown]
# ## HuggingFace WordPiece and special tokens

# %%
# Notebook cell 22
from tokenizers.trainers import WordPieceTrainer

unk_token = "[UNK]"

wordpiece_model = WordPiece(unk_token=unk_token)
wordpiece_tokenizer = Tokenizer(wordpiece_model)
wordpiece_tokenizer.pre_tokenizer = Whitespace()
wordpiece_trainer = WordPieceTrainer(
    vocab_size=28,
    special_tokens=[unk_token]
)

# %%
# Notebook cell 23
wordpiece_tokenizer.train_from_iterator(
    training_data, 
    wordpiece_trainer
)
wordpiece_tokenizer.get_vocab()

# %%
# Notebook cell 24
wordpiece_tokenizer.encode("walker walked a long walk").tokens

# %%
# Notebook cell 25
wordpiece_tokenizer.encode("wlk").tokens

# %%
# Notebook cell 26
wordpiece_tokenizer.encode("she walked").tokens

# %% [markdown]
# ## Unigram

# %%
# Notebook cell 28
from tokenizers.trainers import UnigramTrainer
from tokenizers.models import Unigram

unigram_tokenizer = Tokenizer(Unigram())
unigram_tokenizer.pre_tokenizer = Whitespace()
unigram_trainer = UnigramTrainer(
    vocab_size=14, 
    special_tokens=[unk_token],
    unk_token=unk_token,
)

unigram_tokenizer.train_from_iterator(training_data, unigram_trainer)
unigram_tokenizer.get_vocab()

# %%
# Notebook cell 29
unigram_tokenizer.encode("walker walked a long walk").tokens

# %%
# Notebook cell 30
unigram_tokenizer.encode("wlk").tokens

# %%
# Notebook cell 31
unigram_tokenizer.encode("she walked").tokens

# %%
# Notebook cell 32
unigram_tokenizer.encode("she walked").ids

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

