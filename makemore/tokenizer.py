from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import os

TOKENIZER_PATH = "makemore/bpe_tokenizer.json"

# Step 1: Train the tokenizer if not already trained
def train_bpe_tokenizer(data_file, vocab_size: int = 100):
    if os.path.exists(TOKENIZER_PATH):
        print("Tokenizer already trained. Skipping training.")
        return

    print("Training BPE tokenizer...")
    print(f"Looking for file: {data_file}")  # Debugging statement to verify file path
    assert os.path.exists(data_file), f"File not found: {data_file}"  # Ensure file exists before training

    bpe_tokenizer = Tokenizer(models.BPE())
    bpe_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[],
        initial_alphabet=['0', '1', ',']
    )

    bpe_tokenizer.train(
        files=[data_file],
        trainer=trainer
    )

    bpe_tokenizer.save(TOKENIZER_PATH)
    print(f"Tokenizer saved to {TOKENIZER_PATH}")

# Step 2: Load the tokenizer
# train_bpe_tokenizer()   Will only train if needed
# bpe = Tokenizer.from_file(TOKENIZER_PATH)

# Step 3: Expose encode/decode utilities
# def encode(text: str) -> list[int]:
#     return bpe.encode(text).ids

# def decode(ids: list[int]) -> str:
#     return bpe.decode(ids)

# def vocab_size() -> int:
#     return bpe.get_vocab_size()

# from tokenizers import Tokenizer
# from tokenizers.model import BPE
# from tokenizers.trainers import BpeTrainer
# from tokenizers.pre_tokenizers import Whitespace

# tokenizer = Tokenizer(BPE()) #unk_token="[UNK]"
# tokenizer.pre_tokenizer = Whitespace()

# files = []
# tokenizer.train(files, trainer)
# tokenizer.save

# tokenizer = Tokenizer.from_file()

# output = tokenizer.encode()

if __name__ == "__main__":
    # Example usage
    data_path = "/home/databoss465/poly_lemniscate_project/Data/population2000_deg15_enc.txt"    
    train_bpe_tokenizer(data_path, vocab_size=57)
    bpe = Tokenizer.from_file(TOKENIZER_PATH)
    bitstr = "0,10110010100100,11100001010010,100001100111010,101000100001011,101101001000000,111001010010000,111010011000001,1000010111101001,1000111111101101,1010000101000011,1010100110001110,1011011100101100,1100100101101101,1101000111000001,1111110001100011,1111110010100111,1111110111010010,1111110111010110,1111111001011010"
    tokens = bpe.encode(bitstr).ids
    print(f"Encoded IDs: {tokens}")
    vocab_size = bpe.get_vocab_size()
    print(f"Vocabulary size: {vocab_size}")





