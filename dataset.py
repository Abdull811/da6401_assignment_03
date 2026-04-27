from datasets import load_dataset
import spacy
from collections import Counter
import torch
from torch.nn.utils.rnn import pad_sequence


def load_spacy_tokenizer(model_name, language_code):
    try:
        return spacy.load(model_name)
    except OSError:
        print(
            f"Warning: spaCy model '{model_name}' is not installed. "
            f"Falling back to spacy.blank('{language_code}')."
        )
        return spacy.blank(language_code)

class Multi30kDataset:
    def __init__(
        self,
        split='train',
        src_vocab=None,
        tgt_vocab=None,
        src_itos=None,
        tgt_itos=None,
        min_freq=2,
        max_vocab_size=None,
    ):
        """
        Loads the Multi30k dataset and prepares tokenizers.
        """
        self.split = split
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size
        # Load dataset from Hugging Face
        # https://huggingface.co/datasets/bentrevett/multi30k
        # TODO: Load dataset, load spacy tokenizers for de and en
        self.dataset = load_dataset("bentrevett/multi30k", split=split)

        # load spacy tokenizers
        self.spacy_de = load_spacy_tokenizer("de_core_news_sm", "de")
        self.spacy_en = load_spacy_tokenizer("en_core_web_sm", "en")

        # special tokens
        self.special_tokens = ["<unk>", "<pad>", "<sos>", "<eos>"]

        # vocab placeholders
        self.src_vocab = {}
        self.tgt_vocab = {}

        self.src_itos = []
        self.tgt_itos = []

        # only build vocab on training split
        if split == "train":
            self.build_vocab()
        else:
            self.src_vocab = src_vocab
            self.tgt_vocab = tgt_vocab
            self.src_itos = src_itos
            self.tgt_itos = tgt_itos

        self.processed_data = self.process_data()

    def tokenize_de(self, text):
        return [tok.text.lower() for tok in self.spacy_de.tokenizer(text)]

    def tokenize_en(self, text):
        return [tok.text.lower() for tok in self.spacy_en.tokenizer(text)]

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        return self.processed_data[idx]


    def build_vocab(self):
        """
        Builds the vocabulary mapping for src (de) and tgt (en), including:
        <unk>, <pad>, <sos>, <eos>
        """
        # TODO: Create the vocabulary dictionaries or torchtext Vocab equivalent
        src_counter = Counter()
        tgt_counter = Counter()

        for item in self.dataset:
            src_tokens = self.tokenize_de(item["de"])
            tgt_tokens = self.tokenize_en(item["en"])

            src_counter.update(src_tokens)
            tgt_counter.update(tgt_tokens)

        # start with special tokens
        self.src_itos = self.special_tokens.copy()
        self.tgt_itos = self.special_tokens.copy()

        src_tokens = [
            token for token, freq in src_counter.most_common()
            if freq >= self.min_freq
        ]
        tgt_tokens = [
            token for token, freq in tgt_counter.most_common()
            if freq >= self.min_freq
        ]

        if self.max_vocab_size is not None:
            normal_slots = max(0, self.max_vocab_size - len(self.special_tokens))
            src_tokens = src_tokens[:normal_slots]
            tgt_tokens = tgt_tokens[:normal_slots]

        # add normal words sorted by frequency for reproducibility
        self.src_itos += src_tokens
        self.tgt_itos += tgt_tokens

        self.src_vocab = {
            token: idx for idx, token in enumerate(self.src_itos)
        }

        self.tgt_vocab = {
            token: idx for idx, token in enumerate(self.tgt_itos)
        }

    def process_data(self):
        """
        Convert English and German sentences into integer token lists using
        spacy and the defined vocabulary. 
        """
        # TODO: Tokenize and convert words to indices
        processed = []

        unk_idx = 0
        sos_idx = 2
        eos_idx = 3

        if self.src_vocab is None or self.tgt_vocab is None:
            raise ValueError("Validation/test datasets require src_vocab and tgt_vocab.")

        for item in self.dataset:
            src_tokens = self.tokenize_de(item["de"])
            tgt_tokens = self.tokenize_en(item["en"])

            src_indices = [sos_idx]
            tgt_indices = [sos_idx]

            for token in src_tokens:
                src_indices.append(
                    self.src_vocab.get(token, unk_idx)
                )

            for token in tgt_tokens:
                tgt_indices.append(
                    self.tgt_vocab.get(token, unk_idx)
                )

            src_indices.append(eos_idx)
            tgt_indices.append(eos_idx)

            processed.append((src_indices, tgt_indices))

        return processed


def collate_fn(batch):
    """
    Pads variable-length sequences for DataLoader batching
    """

    src_batch = []
    tgt_batch = []

    for src, tgt in batch:
        src_batch.append(torch.tensor(src))
        tgt_batch.append(torch.tensor(tgt))

    src_batch = pad_sequence(
        src_batch,
        batch_first=True,
        padding_value=1  # <pad>
    )

    tgt_batch = pad_sequence(
        tgt_batch,
        batch_first=True,
        padding_value=1  # <pad>
    )

    return src_batch, tgt_batch
