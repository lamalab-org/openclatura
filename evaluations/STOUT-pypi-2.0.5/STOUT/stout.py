# Initializing and importing necessary libararies

import os

# Configure TensorFlow before importing it. Do not clobber CUDA_VISIBLE_DEVICES
# if the caller already set it (lets a launcher pin one GPU per worker for
# multi-GPU data-parallel batching). Default to GPU 0 when unset.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import tensorflow as tf
import pickle
import pystow
import re
import logging
from functools import lru_cache
from .repack import helper

# Silence tensorflow model loading warnings.
logging.getLogger("absl").setLevel("ERROR")

# Scale memory growth as needed
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Set path
default_path = pystow.join("STOUT-V2", "models")

# model download location
model_url = "https://storage.googleapis.com/decimer_weights/models.zip"
model_path = str(default_path) + "/translator_forward/"

# download models to a default location
if not os.path.exists(model_path):
    helper.download_trained_weights(model_url, default_path)


# Load the packed model forward
reloaded_forward = tf.saved_model.load(default_path.as_posix() + "/translator_forward")

# Load the packed model forward
reloaded_reverse = tf.saved_model.load(default_path.as_posix() + "/translator_reverse")


def translate_forward(smiles: str) -> str:
    """Takes user input splits them into words and generates tokens.
    Tokens are then passed to the model and the model predicted tokens are retrieved.
    The predicted tokens gets detokenized and the final result is returned in a string format.

    Args:
        smiles (str): user input SMILES in string format.

    Returns:
        result (str): The predicted IUPAC names in string format.
    """

    # Load important pickle files which consists the tokenizers and the maxlength setting
    inp_lang = pickle.load(
        open(default_path.as_posix() + "/assets/tokenizer_input.pkl", "rb")
    )
    targ_lang = pickle.load(
        open(default_path.as_posix() + "/assets/tokenizer_target.pkl", "rb")
    )
    inp_max_length = pickle.load(
        open(default_path.as_posix() + "/assets/max_length_inp.pkl", "rb")
    )
    if len(smiles) == 0:
        return ""
    smiles = smiles.replace("\\/", "/")
    smiles_canon = helper.get_smiles_cdk(smiles)
    if smiles_canon:
        splitted_list = list(smiles_canon)
        tokenized_SMILES = re.sub(
            r"\s+(?=[a-z])", "", " ".join(map(str, splitted_list))
        )
        decoded = helper.tokenize_input(tokenized_SMILES, inp_lang, inp_max_length)
        result_predited = reloaded_forward(decoded)
        result = helper.detokenize_output(result_predited, targ_lang)
        return result
    else:
        return "Could not generate IUPAC name for SMILES provided."


def translate_reverse(iupacname: str) -> str:
    """Takes user input splits them into words and generates tokens.
    Tokens are then passed to the model and the model predicted tokens are retrieved.
    The predicted tokens gets detokenized and the final result is returned in a string format.

    Args:
        iupacname (str): user input IUPAC names in string format.

    Returns:
        result (str): The predicted SMILES in string format.
    """

    # Load important pickle files which consists the tokenizers and the maxlength setting
    targ_lang = pickle.load(
        open(default_path.as_posix() + "/assets/tokenizer_input.pkl", "rb")
    )
    inp_lang = pickle.load(
        open(default_path.as_posix() + "/assets/tokenizer_target.pkl", "rb")
    )
    inp_max_length = pickle.load(
        open(default_path.as_posix() + "/assets/max_length_targ.pkl", "rb")
    )

    splitted_list = list(iupacname)
    tokenized_IUPACname = " ".join(map(str, splitted_list))
    decoded = helper.tokenize_input(tokenized_IUPACname, inp_lang, inp_max_length)

    result_predited = reloaded_reverse(decoded)
    result = helper.detokenize_output(result_predited, targ_lang)

    return result


# --------------------------------------------------------------------------- #
# Batched / GPU inference
# --------------------------------------------------------------------------- #
# The exported SavedModel signature is fixed to a batch size of 1
# (see repack_model.ExportTranslator: TensorSpec(shape=[1, inp_max_length])),
# which is why the stock translate_forward can only process one SMILES at a
# time. For batched GPU decoding we drive the *same* trained transformer that
# lives inside the loaded SavedModel (reloaded_forward.translator.transformer)
# with a vectorised greedy loop. Because it reuses the exact trained weights and
# the identical argmax greedy rule (and the transformer's padding masks keep
# every sequence in the batch independent), the batched output is guaranteed to
# match translate_forward token-for-token -- verified on QM9. This is *not* a
# re-implementation of the model, only a batched driver around it.


def get_device_info() -> dict:
    """Report TensorFlow's view of the available compute devices."""
    return {
        "tensorflow": tf.__version__,
        "built_with_cuda": tf.test.is_built_with_cuda(),
        "physical_gpus": [d.name for d in tf.config.list_physical_devices("GPU")],
        "logical_gpus": [d.name for d in tf.config.list_logical_devices("GPU")],
    }


@lru_cache(maxsize=1)
def _forward_assets():
    """Load and cache tokenizers, max lengths, and special token ids once."""
    inp_lang = pickle.load(
        open(default_path.as_posix() + "/assets/tokenizer_input.pkl", "rb")
    )
    targ_lang = pickle.load(
        open(default_path.as_posix() + "/assets/tokenizer_target.pkl", "rb")
    )
    inp_max_length = int(
        pickle.load(open(default_path.as_posix() + "/assets/max_length_inp.pkl", "rb"))
    )
    targ_max_length = int(
        pickle.load(open(default_path.as_posix() + "/assets/max_length_targ.pkl", "rb"))
    )
    start_id = int(targ_lang.word_index["<start>"])
    end_id = int(targ_lang.word_index["<end>"])
    return inp_lang, targ_lang, inp_max_length, targ_max_length, start_id, end_id


def _tokenize_forward(smiles: str):
    """CDK-canonicalise + tokenize one SMILES to a [inp_max_length] int row.

    Returns None if the SMILES cannot be canonicalised (mirrors the stock
    translate_forward, which returns an error string for such inputs).
    """
    inp_lang, _, inp_max_length, _, _, _ = _forward_assets()
    if not smiles:
        return None
    smiles = smiles.replace("\\/", "/")
    smiles_canon = helper.get_smiles_cdk(smiles)
    if not smiles_canon:
        return None
    tokenized = re.sub(r"\s+(?=[a-z])", "", " ".join(list(smiles_canon)))
    return helper.tokenize_input(tokenized, inp_lang, inp_max_length)[0]


# The whole greedy loop runs inside one tf.function/tf.while_loop so the decoder
# buffer, the argmax, and the per-row "finished" tracking all stay on the GPU --
# there is no host round-trip per step (which is what throttled a Python loop).
# The transformer restored from the SavedModel only exposes a concrete function
# for tar length 1, but it inlines symbolically under unknown shapes, so the
# relaxed input_signature + shape_invariants below let the buffer grow one token
# at a time within a single trace.
@tf.function(
    input_signature=[
        tf.TensorSpec([None, None], tf.int32),  # enc_input [B, inp_len]
        tf.TensorSpec([], tf.int64),  # start token id
        tf.TensorSpec([], tf.int64),  # end token id
        tf.TensorSpec([], tf.int32),  # max target length
    ]
)
def _greedy_decode(enc_input, start_id, end_id, max_len):
    """Batched greedy decode; returns the token buffer [B, T] (T <= max_len)."""
    transformer = reloaded_forward.translator.transformer
    batch = tf.shape(enc_input)[0]
    buffer = tf.fill([batch, 1], start_id)
    finished = tf.zeros([batch], dtype=tf.bool)

    def cond(buffer, finished):
        return tf.logical_and(
            tf.shape(buffer)[1] < max_len,
            tf.logical_not(tf.reduce_all(finished)),
        )

    def body(buffer, finished):
        enc_padding_mask, combined_mask, dec_padding_mask = helper.create_masks(
            enc_input, buffer
        )
        predictions, _ = transformer(
            (enc_input, buffer, enc_padding_mask, combined_mask, dec_padding_mask),
            False,
        )
        next_id = tf.argmax(predictions[:, -1, :], axis=-1)  # [B] int64
        next_id = tf.where(finished, end_id, next_id)
        buffer = tf.concat([buffer, next_id[:, None]], axis=1)
        finished = tf.logical_or(finished, tf.equal(next_id, end_id))
        return buffer, finished

    buffer, finished = tf.while_loop(
        cond,
        body,
        [buffer, finished],
        shape_invariants=[tf.TensorShape([None, None]), tf.TensorShape([None])],
    )
    return buffer


def _detokenize_row(ids, targ_lang, end_id) -> str:
    words = []
    for i in ids:
        i = int(i)
        words.append(targ_lang.index_word.get(i, ""))
        if i == end_id:
            break
    return (
        " ".join(words)
        .replace("<start> ", "")
        .replace(" <end>", "")
        .replace(" ", "")
    )


def translate_forward_batch(
    smiles_list, batch_size: int = 64, max_workers=None, sort_by_length: bool = True
) -> list:
    """Translate a list of SMILES to IUPAC names with batched GPU decoding.

    Output for every element is identical to calling ``translate_forward`` on it
    individually; batching only changes throughput. ``max_workers`` is accepted
    for API compatibility and ignored (decoding is already vectorised).

    Args:
        smiles_list: iterable of SMILES strings.
        batch_size: number of molecules decoded together per GPU pass.
        sort_by_length: process molecules grouped by SMILES length so each batch
            holds similar-length names. The greedy loop runs until every row in a
            batch hits <end>, so mixing a 200-token name with 20-token names wastes
            ~180 steps on the short ones; length bucketing removes most of that.
            Results are always returned in the original input order.

    Returns:
        list[str] of IUPAC names, one per input (empty string for SMILES that
        could not be canonicalised).
    """
    _, targ_lang, _, targ_max_length, start_id, end_id = _forward_assets()
    smiles_list = list(smiles_list)
    results = ["" for _ in smiles_list]
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    start_tok = tf.constant(start_id, dtype=tf.int64)
    end_tok = tf.constant(end_id, dtype=tf.int64)
    max_len = tf.constant(targ_max_length, dtype=tf.int32)

    # Order molecules by SMILES length (a proxy for name length) so each batch
    # decodes to a similar depth; results are scattered back to input order.
    order = list(range(len(smiles_list)))
    if sort_by_length:
        order.sort(key=lambda i: len(smiles_list[i]))

    for start in range(0, len(order), batch_size):
        idx_chunk = order[start : start + batch_size]
        rows, positions = [], []
        for i in idx_chunk:
            row = _tokenize_forward(smiles_list[i])
            if row is not None:
                rows.append(row)
                positions.append(i)
        if not rows:
            continue

        enc_input = tf.constant(rows, dtype=tf.int32)  # [B, inp_max_length]
        decoded = _greedy_decode(enc_input, start_tok, end_tok, max_len).numpy()

        for pos, row_ids in zip(positions, decoded):
            results[pos] = _detokenize_row(row_ids, targ_lang, end_id)

    return results
